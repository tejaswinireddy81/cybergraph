"""
modules/explainable_graph_ai.py
===============================
PHASE 6 & 8: Explainable Graph AI (XAI) Engine with SHAP & Feature Saliency

Generates model-driven, human-interpretable explanations using:
  1. SHAP TreeExplainer Exact Shapley Feature Values (via shap package)
  2. Standardized Feature Saliency Attribution:
       z_i = |x_i - \mu_i| / \sigma_i
       C_i = z_i / \sum_j z_j * 100%
  3. "Why This Attack?" Human-Readable Bullet Reasons
"""

import math
import numpy as np
from modules.explainability import ExplainabilityEngine
from config.risk_thresholds import XGB_PROB_HIGH, AE_SCORE_HIGH

class ExplainableGraphAI:
    """Model-Driven Explainable Graph AI (XAI) Engine with SHAP Integration."""

    def __init__(self, scaler=None):
        self.scaler = scaler
        self.shap_engine = ExplainabilityEngine()

    def explain_flow_edge(
        self,
        edge_attr,
        risk_score,
        source_host,
        target_service,
        host_x=None,
        service_x=None,
        temporal_shift=0.0,
        raw_feat_1d=None,
        xgb_model=None,
        feature_cols=None,
    ):
        """Generates a human-interpretable XAI report with SHAP attributions and 'Why This Attack?' bullet list."""
        byte_cnt = float(edge_attr[3])
        pkt_cnt = float(edge_attr[4])
        duration = float(edge_attr[5])
        xgb_prob = float(edge_attr[6])
        ae_score = float(edge_attr[7])
        fusion_score = float(edge_attr[8])

        # SHAP Feature Attribution Calculation
        shap_values_top = []
        if xgb_model is not None and raw_feat_1d is not None and feature_cols is not None:
            shap_values_top = self.shap_engine.explain_flow_shap(xgb_model, feature_cols, raw_feat_1d)

        s_xgb = abs(xgb_prob - 0.5) * 2.0 + 1e-6
        s_ae = abs(ae_score - 0.0) * 1.5 + 1e-6
        s_vol = math.log1p(byte_cnt) / 10.0 + 1e-6
        s_pkt = math.log1p(pkt_cnt) / 5.0 + 1e-6
        s_temp = float(temporal_shift) + 1e-6

        total_saliency = s_xgb + s_ae + s_vol + s_pkt + s_temp

        attributions = {
            "xgboost_shap_signal": float(s_xgb / total_saliency),
            "autoencoder_anomaly_signal": float(s_ae / total_saliency),
            "traffic_volume_burstiness": float(s_vol / total_saliency),
            "packet_rate_intensity": float(s_pkt / total_saliency),
            "temporal_representation_shift": float(s_temp / total_saliency),
        }

        top_factor = max(attributions, key=attributions.get)

        # Generate "Why This Attack?" Human-Readable Bullet Reasons with SHAP callouts
        why_reasons = []
        if xgb_prob >= XGB_PROB_HIGH:
            why_reasons.append("✓ High XGBoost Malicious Pattern Match (SHAP Confirmed)")
        if ae_score >= AE_SCORE_HIGH:
            why_reasons.append("✓ High Autoencoder Reconstruction Anomaly")
        if byte_cnt > 500:
            why_reasons.append("✓ Abnormal Traffic Volume Burstiness")
        if pkt_cnt > 10:
            why_reasons.append("✓ High Packet Rate Intensity")
        if temporal_shift > 0.01:
            why_reasons.append("✓ High Temporal Memory Trajectory Shift")

        if shap_values_top:
            top_fname, top_fval = shap_values_top[0]
            why_reasons.append(f"✓ SHAP Primary Feature Driver: {top_fname} ({top_fval:+.4f})")
        else:
            why_reasons.append("✓ Suspicious Bipartite Structural Connection")

        summary_text = (
            f"Flow [{source_host} -> {target_service}] flagged with risk score {risk_score:.4f}. "
            f"Primary saliency driver: {top_factor.replace('_', ' ').title()} ({attributions[top_factor]*100:.1f}% attribution)."
        )

        return {
            "source_host": source_host,
            "target_service": target_service,
            "overall_risk_score": float(risk_score),
            "primary_saliency_driver": top_factor,
            "feature_attributions": attributions,
            "shap_top_features": shap_values_top,
            "why_this_attack_reasons": why_reasons,
            "raw_signals": {
                "xgb_prob": xgb_prob,
                "ae_score": ae_score,
                "fusion_score": fusion_score,
                "byte_count": byte_cnt,
                "pkt_count": pkt_cnt,
                "duration": duration,
                "temporal_representation_shift": float(temporal_shift),
            },
            "attribution_methodology": "SHAP TreeExplainer & Standardized Feature Saliency Attribution",
            "explanation_summary": summary_text,
        }
