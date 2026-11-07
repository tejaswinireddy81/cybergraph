"""
Unit tests for modules/fusion_engine.py verifying fallback vs trained-model paths.
"""

import pytest
import os
import joblib
from modules.fusion_engine import FusionEngine


def test_fusion_engine_fallback_mode(monkeypatch, tmp_path):
    # Point _MODEL_PATH to non-existent path to force fallback mode
    fake_path = str(tmp_path / "non_existent_fusion_model.pkl")
    monkeypatch.setattr("modules.fusion_engine._MODEL_PATH", fake_path)

    engine = FusionEngine()
    assert engine.using_fallback is True

    # Test fixed weight formula: rf*0.40 + ae*0.20 + gat*0.40
    # For rf=1.0, ae=1.0, gat=1.0 -> score = 1.0
    res = engine.predict(rf=1.0, ae=1.0, gat=1.0)
    assert res["score"] == 1.0
    assert res["risk"] == "High Risk"
    assert res["using_fallback_weights"] is True

    # Medium risk range (0.40 to 0.69)
    res_med = engine.predict(rf=0.5, ae=0.5, gat=0.5)
    assert res_med["score"] == 0.5
    assert res_med["risk"] == "Medium Risk"

    # Low risk range (< 0.40)
    res_low = engine.predict(rf=0.1, ae=0.1, gat=0.1)
    assert res_low["score"] == 0.1
    assert res_low["risk"] == "Low Risk"


class MockModel:
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.15, 0.85]])


def test_fusion_engine_trained_model(monkeypatch, tmp_path):
    mock_model_path = str(tmp_path / "fusion_model.pkl")
    joblib.dump(MockModel(), mock_model_path)
    monkeypatch.setattr("modules.fusion_engine._MODEL_PATH", mock_model_path)

    engine = FusionEngine()
    assert engine.using_fallback is False

    res = engine.predict(rf=0.8, ae=0.6, gat=0.9)
    assert res["score"] == 0.85
    assert res["risk"] == "High Risk"
    assert res["using_fallback_weights"] is False
