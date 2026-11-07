"""
fusion_engine.py
=================
PHASE 1 RECONCILIATION NOTE:
This module (FusionEngine) is the LEGACY fusion implementation that expects
3 inputs: (rf_prob, ae_score, gat_score) — matching the Generation 1 pipeline
(Random Forest + Keras AE + Static GAT).

CANONICAL LIVE PIPELINE (app.py / LiveTrafficEngine) does NOT use FusionEngine.
The canonical fusion is implemented inline in modules/live_traffic_engine.py:
  - 2 inputs: [xgb_prob, ae_norm]
  - Trained logistic regression meta-learner in models/fusion_model.pkl
  - Written by train_xgboost_ae.py

FusionEngine / train_fusion.py write a 3-feature model to models/fusion_model.pkl
which CONFLICTS with the 2-feature model written by train_xgboost_ae.py.
This disambiguation is handled by keeping this module in the codebase as legacy
documentation, while the canonical pipeline uses live_traffic_engine.py exclusively.

If you run train_fusion.py (3-feature), it will break LiveTrafficEngine.
If you run train_xgboost_ae.py (2-feature), it will break inference.py (legacy).
Both are intentionally preserved for historical reference.

See legacy/rf_pipeline/inference.py for the Generation 1 use of FusionEngine.
"""
import os
import joblib
import numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fusion_model.pkl")

# PHASE 1 NOTE: These weights are for the LEGACY pipeline (rf/ae/gat).
# The canonical pipeline uses a learned 2-feature logistic regression (xgb, ae).
_FALLBACK_WEIGHTS = {"rf": 0.40, "ae": 0.20, "gat": 0.40}  # used only if no trained fusion model exists


class FusionEngine:
    """
    LEGACY 3-signal fusion: (rf_prob, ae_score, gat_score).

    WARNING: This class is used by legacy/rf_pipeline/inference.py only.
    The canonical live pipeline (app.py → LiveTrafficEngine) implements
    its own 2-signal fusion inline and does NOT import or call FusionEngine.

    Do NOT use FusionEngine in the canonical pipeline.
    """
    def __init__(self):
        self.model = None
        self.using_fallback = True
        if os.path.exists(_MODEL_PATH):
            self.model = joblib.load(_MODEL_PATH)
            self.using_fallback = False

    def calculate_score(self, rf, ae, gat):
        """
        Compute fused attack probability from 3 legacy signals.

        Args:
            rf  (float): Random Forest attack probability [0, 1]
            ae  (float): Normalized Autoencoder reconstruction error [0, 1]
            gat (float): Static GAT host-graph attack score [0, 1]

        Returns:
            float: Combined attack probability [0, 1]

        Note: In the CANONICAL pipeline, the equivalent call is:
              LogisticRegression([[xgb_prob, ae_norm]]).predict_proba()
              handled by LiveTrafficEngine.process_flow_record().
        """
        if self.model is not None:
            X = np.array([[rf, ae, gat]])
            return float(self.model.predict_proba(X)[0, 1])
        # fallback: fixed weights, only used before train_fusion.py has been run
        return (rf * _FALLBACK_WEIGHTS["rf"] +
                ae * _FALLBACK_WEIGHTS["ae"] +
                gat * _FALLBACK_WEIGHTS["gat"])

    def predict(self, rf, ae, gat):
        score = self.calculate_score(rf, ae, gat)
        if score >= 0.70:
            label = "High Risk"
        elif score >= 0.40:
            label = "Medium Risk"
        else:
            label = "Low Risk"
        return {
            "score": round(score, 4),
            "risk": label,
            "using_fallback_weights": self.using_fallback,
        }
