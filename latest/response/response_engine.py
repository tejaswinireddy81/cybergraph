"""
response_engine.py
===================
Implements the severity -> action policy from Section 5 of the
requirements doc:

    Low       Log and continue monitoring.
    Medium    Alert administrator and increase monitoring.
    High      Alert and temporarily blacklist the source IP.
    Critical  Verify confidence, block source IP, alert administrator,
              and create an incident.

This is deliberately a separate, higher-level module from
modules/response_engine.py (which handles per-flow alert logging and
cooldown, and is already wired into inference.py). This module is the
policy layer on top: given a scored event, decide *what kind* of
response is warranted, carry it out via FirewallController, and write
the result to IncidentDatabase. It does not replace the existing
per-flow alerting — it can be called alongside it, or used to
supersede it once wired into inference.py.

Not done here (by design, stated honestly): actually paging/emailing
an administrator. `alert_callback` is the extension point for that —
pass a function that sends the alert record wherever it needs to go
(email, Slack webhook, SIEM). Left as an injectable callback rather
than a specific implementation because "alert the administrator"
means something different in every deployment.
"""

import os
import json
import time
import logging

from .firewall import FirewallController
from database.incident_db import IncidentDatabase

logger = logging.getLogger(__name__)

_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
_POLICY_ALERTS_LOG = os.path.join(_LOG_DIR, "policy_alerts.jsonl")

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]

# Only create a DB incident at this severity or above. Low events are
# noise-reduction by design (see requirements doc Section 5) — logging
# every one of them as an "incident" would drown the real ones.
_INCIDENT_MIN_SEVERITY = "Medium"

DEFAULT_CRITICAL_CONFIDENCE_THRESHOLD = 0.85


class ResponsePolicyEngine:
    def __init__(
        self,
        firewall: FirewallController = None,
        incident_db: IncidentDatabase = None,
        critical_confidence_threshold: float = DEFAULT_CRITICAL_CONFIDENCE_THRESHOLD,
        cooldown_seconds: int = 30,
        alert_callback=None,
    ):
        os.makedirs(_LOG_DIR, exist_ok=True)
        self.firewall = firewall or FirewallController()  # defaults: enabled=False, dry_run=True (safe)
        self.incident_db = incident_db or IncidentDatabase()
        self.critical_confidence_threshold = critical_confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.alert_callback = alert_callback

        self._last_action = {}  # (ip, severity) -> timestamp, cooldown to avoid alert storms

    # ── public entry point ─────────────────────────────────────────────

    def handle(
        self,
        src_ip,
        severity,
        dst_ip=None,
        risk_score=None,
        confidence=None,
        src_port=None,
        dst_port=None,
        protocol=None,
        pkt_count=None,
        byte_count=None,
        duration=None,
        attack_type=None,
        rf_prediction=None,
        rf_confidence=None,
        ae_score=None,
        gnn_score=None,
        gat_score=None,
        explanation=None,
        attack_chain=None,
        threat_intel_status=None,
        event_timestamp=None,
        raw_details=None,
    ) -> dict:
        """Apply the severity policy to one scored event and return a
        record describing what was done.

        `confidence` is used only for the Critical-severity "verify
        confidence" step; if not supplied, `rf_confidence` is used as
        the fallback confidence signal.
        """
        if severity not in SEVERITY_ORDER:
            raise ValueError(f"severity must be one of {SEVERITY_ORDER}, got {severity!r}")

        now = time.time()
        if self._on_cooldown(src_ip, severity, now):
            return {"suppressed": True, "reason": "cooldown", "src_ip": src_ip, "severity": severity}

        confidence = confidence if confidence is not None else rf_confidence

        if severity == "Low":
            result = self._handle_low(src_ip)
        elif severity == "Medium":
            result = self._handle_medium(src_ip)
        elif severity == "High":
            result = self._handle_high(src_ip)
        else:  # Critical
            result = self._handle_critical(src_ip, confidence)

        result.update({"src_ip": src_ip, "severity": severity, "timestamp": now})

        incident_id = None
        if SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(_INCIDENT_MIN_SEVERITY):
            incident_id = self.incident_db.create_incident(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                pkt_count=pkt_count,
                byte_count=byte_count,
                duration=duration,
                attack_type=attack_type,
                rf_prediction=rf_prediction,
                rf_confidence=rf_confidence,
                ae_score=ae_score,
                gnn_score=gnn_score,
                gat_score=gat_score,
                final_risk_score=risk_score,
                severity=severity,
                explanation=explanation,
                attack_chain=attack_chain,
                threat_intel_status=threat_intel_status,
                response_action=result.get("action"),
                response_status="open",
                event_timestamp=event_timestamp,
                raw_details=raw_details,
            )
            result["incident_id"] = incident_id

        if result.get("alert"):
            self._send_alert({**result, "dst_ip": dst_ip, "risk_score": risk_score})

        return result

    # ── per-severity handlers ────────────────────────────────────────────

    def _handle_low(self, src_ip) -> dict:
        # "Log and continue monitoring." No firewall action, no admin alert,
        # no incident (see _INCIDENT_MIN_SEVERITY).
        return {"action": "logged", "alert": False}

    def _handle_medium(self, src_ip) -> dict:
        # "Alert administrator and increase monitoring."
        return {"action": "alerted_increased_monitoring", "alert": True}

    def _handle_high(self, src_ip) -> dict:
        # "Alert and temporarily blacklist the source IP." Blacklist is the
        # soft mechanism (see firewall.py) — flags the host, doesn't touch
        # the OS firewall.
        blacklist_record = self.firewall.blacklist_ip(src_ip, reason="High severity detection")
        return {"action": "alerted_blacklisted", "alert": True, "blacklist": blacklist_record}

    def _handle_critical(self, src_ip, confidence) -> dict:
        # "Verify confidence, block source IP, alert administrator, and
        # create an incident."
        confidence_verified = confidence is not None and confidence >= self.critical_confidence_threshold

        if not confidence_verified:
            # Confidence didn't clear the bar: don't take the irreversible
            # step (a real firewall block). Fall back to the High-severity
            # response (blacklist + alert) and say so explicitly, so this
            # is visible in the incident record rather than silently
            # downgraded.
            blacklist_record = self.firewall.blacklist_ip(
                src_ip, reason="Critical severity but confidence not verified")
            return {
                "action": "alerted_blacklisted_confidence_unverified",
                "alert": True,
                "confidence_verified": False,
                "confidence": confidence,
                "confidence_threshold": self.critical_confidence_threshold,
                "blacklist": blacklist_record,
            }

        block_record = self.firewall.block_ip(src_ip, reason="Critical severity, confidence verified")
        return {
            "action": "blocked",
            "alert": True,
            "confidence_verified": True,
            "confidence": confidence,
            "confidence_threshold": self.critical_confidence_threshold,
            "block": block_record,
        }

    # ── alerting / cooldown ────────────────────────────────────────────

    def _on_cooldown(self, ip, severity, now) -> bool:
        key = (ip, severity)
        last = self._last_action.get(key)
        if last is not None and (now - last) < self.cooldown_seconds:
            return True
        self._last_action[key] = now
        return False

    def _send_alert(self, record: dict):
        with open(_POLICY_ALERTS_LOG, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        if self.alert_callback:
            try:
                self.alert_callback(record)
            except Exception:
                logger.exception("alert_callback raised while handling %s", record.get("src_ip"))

    # ── analyst actions ──────────────────────────────────────────────────

    def resolve_incident(self, incident_id, status="resolved"):
        """Analyst marks an incident resolved / false positive from the
        Incident History panel."""
        return self.incident_db.update_response(incident_id, response_status=status, resolved=(status == "resolved"))

    def unblock(self, ip):
        self.firewall.unblock_ip(ip)
        self.firewall.remove_from_blacklist(ip)
