"""
response_engine.py
===================
Turns a fused risk score into an actual alert/response action. This was
an empty (0-byte) stub in the original scaffold.

Scope, stated honestly: this simulates response actions (structured
alert logging + a "blocked hosts" list) rather than actually integrating
with a real firewall/SDN controller. Wiring it to an actual enforcement
point (iptables, a switch ACL, a cloud security group) is a natural next
step, but claiming that integration without building and testing it
against a real device would be dishonest, so it isn't included here.
"""
import os
import json
import time

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
ALERTS_PATH = os.path.join(LOG_DIR, "alerts.jsonl")

COOLDOWN_SECONDS = 30  # don't re-alert on the same host+level within this window


class ResponseEngine:
    def __init__(self, cooldown_seconds=COOLDOWN_SECONDS):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.cooldown_seconds = cooldown_seconds
        self._last_alert = {}          # (ip, level) -> timestamp
        self.blocked_hosts = set()      # simulated block list

    def _on_cooldown(self, ip, level, now):
        key = (ip, level)
        last = self._last_alert.get(key)
        if last is not None and (now - last) < self.cooldown_seconds:
            return True
        self._last_alert[key] = now
        return False

    def handle_threat(self, src_ip, dst_ip, risk_score, risk_level, details=None):
        """Call this once per scored flow/host. Returns the action taken
        (or None if suppressed by cooldown)."""
        now = time.time()
        if risk_level == "Low":
            return None
        if self._on_cooldown(src_ip, risk_level, now):
            return None

        action = "logged"
        if risk_level == "Critical" and src_ip not in self.blocked_hosts:
            self.blocked_hosts.add(src_ip)
            action = "simulated_block"

        record = {
            "timestamp": now,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "action": action,
            "details": details or {},
        }
        with open(ALERTS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def is_blocked(self, ip):
        return ip in self.blocked_hosts

    def unblock(self, ip):
        self.blocked_hosts.discard(ip)
