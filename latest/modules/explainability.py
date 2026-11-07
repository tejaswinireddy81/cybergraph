"""
explainability.py
=================
EXPLAINABLE AI ENGINE (SHAP TREEEXPLAINER INTEGRATION)

Provides exact SHAP feature attributions for XGBoost model predictions.
Computes feature-level shapley values:
  - Exact feature attributions for top suspicious signals
  - Direct integration into CYBERGRAPH COMMAND CENTER threat feed
"""

import shap
import numpy as np


class ExplainabilityEngine:
    """SHAP TreeExplainer Engine for Model Explainability."""

    def __init__(self):
        self.shap_explainer = None

    def load(self, xgb_model):
        """Loads SHAP TreeExplainer for the trained XGBoost classifier."""
        try:
            self.shap_explainer = shap.TreeExplainer(xgb_model)
        except Exception:
            self.shap_explainer = None

    def explain_flow_shap(self, xgb_model, feature_names, sample_1d):
        """Computes top 5 SHAP feature contributions for a single flow sample."""
        if self.shap_explainer is None:
            self.load(xgb_model)

        if self.shap_explainer is None:
            return []

        try:
            sample_2d = np.array([sample_1d], dtype=np.float32)
            shap_vals = self.shap_explainer.shap_values(sample_2d)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[0]

            top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]
            results = []
            for i in top_idx:
                fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                val = float(shap_vals[i])
                results.append((fname, val))

            return results
        except Exception:
            return []