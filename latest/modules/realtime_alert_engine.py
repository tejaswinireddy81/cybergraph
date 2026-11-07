"""
modules/realtime_alert_engine.py
================================
PHASE 3: Tiered Real-Time Security Response & Alerting Engine with SHAP Integration
"""

import time
import logging

from config.risk_thresholds import (
    RISK_L1_MAX,
    RISK_L2_MAX,
    BARI_L1_MAX,
    BARI_L2_MAX,
    XGB_PROB_LOW,
)

from modules.attack_path_predictor import AttackPathPredictor
from modules.explainable_graph_ai import ExplainableGraphAI
from modules.firewall_response_executor import FirewallResponseExecutor
from modules.attack_journey_engine import AttackJourneyEngine
from train_xgboost_ae import FEATURE_COLS


logger = logging.getLogger(__name__)


class RealtimeAlertEngine:
    """Tiered real-time security response and alerting engine."""

    def __init__(self, enable_real_blocking=False, bari_engine=None):
        self.enable_real_blocking = enable_real_blocking
        self.predictor = AttackPathPredictor(risk_threshold=0.50)
        self.xai = ExplainableGraphAI()
        self.fw_executor = FirewallResponseExecutor(
            enable_real_blocking=enable_real_blocking
        )
        self.journey_engine = AttackJourneyEngine()
        self.bari_engine = bari_engine

    def _safe_xai_explain(
        self,
        edge_attr,
        risk_score,
        source_host,
        target_service,
        raw_feat_1d,
        xgb_model,
        feature_cols,
    ):
        """Run XAI without allowing XAI failure to stop alert processing."""
        try:
            return self.xai.explain_flow_edge(
                edge_attr=edge_attr,
                risk_score=risk_score,
                source_host=source_host,
                target_service=target_service,
                raw_feat_1d=raw_feat_1d,
                xgb_model=xgb_model,
                feature_cols=feature_cols,
            )
        except Exception as exc:
            logger.exception("XAI explanation failed; continuing alert pipeline: %s", exc)
            return {
                "status": "XAI_UNAVAILABLE",
                "message": "XAI explanation unavailable; alert processing continued.",
                "error": str(exc),
            }

    def process_flow_eval(
        self,
        flow_rec,
        risk_score,
        snapshot=None,
        host_states=None,
        xgb_model=None,
        bari_res=None,
        node_type_resolver=None,
    ):
        """Process a flow and return the corresponding security response."""

        risk_score = float(risk_score)

        src_ip = flow_rec["src_ip"]
        dst_ip = flow_rec["dst_ip"]
        dst_port = flow_rec["dst_port"]
        proto = flow_rec["protocol"]

        ae_score = float(flow_rec.get("ae_score", 0.0))
        xgb_prob = float(flow_rec.get("xgb_prob", 0.0))
        raw_feat_1d = flow_rec.get("raw_features", None)

        bari_score = 0.0
        route_str = ""
        exp_summary = ""

        if isinstance(bari_res, dict):
            bari_score = float(bari_res.get("bari_score", 0.0))
            route_str = bari_res.get("route_string", "")
            exp_summary = bari_res.get("explanation_summary", "")
        elif isinstance(bari_res, (float, int)):
            bari_score = float(bari_res)

        if node_type_resolver:
            ntype = node_type_resolver(src_ip)
            host_id = f"{ntype}:{src_ip}"
        else:
            host_id = f"HOST:{src_ip}"
        service_id = f"SERVICE:{dst_port}/{proto}"

        base_ml_evidence = (
    xgb_prob >= 0.70
    or ae_score >= 0.50
    or risk_score >= RISK_L1_MAX)
        if not base_ml_evidence:
            level_name = "Level 1 — Normal"
        elif risk_score < RISK_L2_MAX and bari_score < BARI_L2_MAX:
            level_name = "Level 2 — Suspicious / Emerging"
        else:
            level_name = "Level 3 — High Risk"

        journey_payload = self.journey_engine.process_flow_journey(
            src_ip=src_ip,
            dst_port=dst_port,
            protocol=proto,
            risk_score=risk_score,
            bari_res=bari_res if isinstance(bari_res, dict) else {},
            level_str=level_name,
            timestamp=flow_rec.get("timestamp", time.time()),
            bari_engine=self.bari_engine,
        )

        if level_name == "Level 1 — Normal":
            return {
                "level": "Level 1 — Normal",
                "risk_category": "BENIGN",
                "risk_score": risk_score,
                "bari_score": bari_score,
                "action": "ALLOW_AND_MONITOR",
                "source_host": host_id,
                "target_service": service_id,
                "timestamp": flow_rec["timestamp"],
                "badge_color": "#10b981",
                "summary": (
                    f"Flow [{host_id} -> {service_id}] benign "
                    f"(Risk: {risk_score:.4f}, BARI: {bari_score:.4f}). Allowed."
                ),
                "attack_journey": journey_payload,
            }

        edge_attr = [
            flow_rec["timestamp"],
            flow_rec["dst_port"],
            flow_rec["protocol"],
            flow_rec["byte_count"],
            flow_rec["pkt_count"],
            flow_rec["duration"],
            flow_rec["xgb_prob"],
            flow_rec["ae_score"],
            flow_rec["fusion_score"],
        ]

        xai_report = self._safe_xai_explain(
            edge_attr=edge_attr,
            risk_score=risk_score,
            source_host=host_id,
            target_service=service_id,
            raw_feat_1d=raw_feat_1d,
            xgb_model=xgb_model,
            feature_cols=FEATURE_COLS,
        )

        if level_name == "Level 2 — Suspicious / Emerging":
            return {
                "level": "Level 2 — Suspicious / Emerging",
                "risk_category": "EARLY_WARNING_THREAT",
                "risk_score": risk_score,
                "bari_score": bari_score,
                "action": "EARLY_WARNING_SOC_ALERT",
                "source_host": host_id,
                "target_service": service_id,
                "timestamp": flow_rec["timestamp"],
                "badge_color": "#f59e0b",
                "xai_explanation": xai_report,
                "route_explanation": (
                    exp_summary or f"Observed Route: {route_str}"
                ),
                "summary": (
                    f"EARLY WARNING: [{host_id} -> {service_id}] "
                    f"Route Anomaly Detected "
                    f"(BARI: {bari_score:.4f}, Risk: {risk_score:.4f}). "
                    f"{exp_summary}"
                ),
                "attack_journey": journey_payload,
            }

        if not base_ml_evidence or risk_score < RISK_L2_MAX:
            logger.warning(
        "[BLOCK SUPPRESSED] src=%s dst_port=%s "
        "risk=%.6f bari=%.6f base_ml=%s",
        src_ip,
        dst_port,
        risk_score,
        bari_score,
        base_ml_evidence,
    )
            return {
        "level": "Level 2 — Suspicious / Emerging",
        "risk_category": "EARLY_WARNING_THREAT",
        "risk_score": risk_score,
        "bari_score": bari_score,
        "action": "EARLY_WARNING_SOC_ALERT",
        "source_host": host_id,
        "target_service": service_id,
        "timestamp": flow_rec["timestamp"],
        "badge_color": "#f59e0b",
        "xai_explanation": xai_report,
        "route_explanation": (
            exp_summary or f"Observed Route: {route_str}"
        ),
        "summary": (
            f"EARLY WARNING: [{host_id} -> {service_id}] "
            f"Contextual anomaly detected "
            f"(BARI: {bari_score:.4f}, Risk: {risk_score:.4f})."
        ),
        "attack_journey": journey_payload,
    }
        fw_res = self.fw_executor.execute_block(
    src_ip,
    port=dst_port,
)

        attack_paths = []

        if snapshot is not None:
            attack_paths = self.predictor.predict_attack_paths(
                snapshot,
                top_k_paths=1,
            )

        dst_node_label = f"{node_type_resolver(dst_ip)}:{dst_ip}" if node_type_resolver else f"EXTERNAL:{dst_ip}"
        bipartite_chain = (
            attack_paths[0]["bipartite_attack_chain"]
            if attack_paths
            else [
                host_id,
                service_id,
                dst_node_label,
                "SERVICE:80/6",
                "EXTERNAL:10.40.85.10",
            ]
        )

        return {
            "level": "Level 3 — High Risk",
            "risk_category": "CRITICAL_ATTACK_THREAT",
            "risk_score": risk_score,
            "action": "CRITICAL_ALERT_RECOMMENDED_BLOCK",
            "source_host": host_id,
            "target_service": service_id,
            "timestamp": flow_rec["timestamp"],
            "badge_color": "#ef4444",
            "xai_explanation": xai_report,
            "bipartite_attack_chain": bipartite_chain,
            "recommended_firewall_action": {
                "status": fw_res["status"],
                "execution_mode": fw_res["execution_mode"],
                "target_ip": src_ip,
                "target_port": dst_port,
                "windows_netsh_rule_preview": fw_res.get(
                    "command_preview",
                    fw_res.get("message", ""),
                ),
                "real_blocking_enabled": self.enable_real_blocking,
            },
            "summary": (
                f"CRITICAL ATTACK: [{host_id} -> {service_id}] "
                f"Risk: {risk_score:.4f}. "
                f"Recommended Action: Block IP {src_ip}."
            ),
            "attack_journey": journey_payload,
        }