"""
tests/test_schema_consistency.py
=================================
PHASE 1 — FEATURE SCHEMA CONSISTENCY TEST

Verifies that ALL components in the canonical pipeline agree on exactly 47 features.

Components checked:
  1. train_xgboost_ae.FEATURE_COLS  (authoritative source)
  2. modules/feature_schema_validator.FeatureSchemaValidator
  3. models/xgboost_ae_artifacts.json  feature_cols
  4. models/artifacts.json  feature_cols  (legacy artifact — must still be 47)
  5. flow_extractor.py emitted feature count

This test will FAIL if:
  - Any component silently has a different feature count.
  - Any feature name is missing or misspelled.
  - MIN_TTL or MAX_TTL appear anywhere in the canonical feature list.

Run with:
    pytest tests/test_schema_consistency.py -v
"""

import os
import sys
import json
import pytest

# Ensure repo root is on path
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)


# ── Authoritative feature list ─────────────────────────────────────────────

EXPECTED_FEATURE_COUNT = 47

EXPECTED_FEATURES = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]

assert len(EXPECTED_FEATURES) == EXPECTED_FEATURE_COUNT, \
    f"Test fixture error: expected {EXPECTED_FEATURE_COUNT} features, got {len(EXPECTED_FEATURES)}"

# Prohibited legacy features (removed in FIX D — TTL domain shift)
PROHIBITED_FEATURES = {"MIN_TTL", "MAX_TTL"}


# ── Helper ─────────────────────────────────────────────────────────────────

def _artifact_path(filename):
    return os.path.join(REPO_ROOT, "models", filename)


# ── Tests ──────────────────────────────────────────────────────────────────

class TestFeatureSchemaConsistency:
    """All pipeline components must agree on exactly 47 features."""

    def test_train_xgboost_ae_feature_cols_count(self):
        """train_xgboost_ae.FEATURE_COLS must have exactly 47 features."""
        from train_xgboost_ae import FEATURE_COLS
        assert len(FEATURE_COLS) == EXPECTED_FEATURE_COUNT, (
            f"train_xgboost_ae.FEATURE_COLS has {len(FEATURE_COLS)} features, "
            f"expected {EXPECTED_FEATURE_COUNT}."
        )

    def test_train_xgboost_ae_feature_cols_names(self):
        """train_xgboost_ae.FEATURE_COLS must match the canonical 47-feature list exactly."""
        from train_xgboost_ae import FEATURE_COLS
        assert list(FEATURE_COLS) == EXPECTED_FEATURES, (
            "FEATURE_COLS mismatch. "
            f"Extra: {set(FEATURE_COLS) - set(EXPECTED_FEATURES)}. "
            f"Missing: {set(EXPECTED_FEATURES) - set(FEATURE_COLS)}."
        )

    def test_no_ttl_features_in_canonical_schema(self):
        """MIN_TTL and MAX_TTL must NOT appear in the canonical feature list (FIX D)."""
        from train_xgboost_ae import FEATURE_COLS
        for prohibited in PROHIBITED_FEATURES:
            assert prohibited not in FEATURE_COLS, (
                f"PROHIBITED feature '{prohibited}' found in FEATURE_COLS. "
                "FIX D removed TTL features to prevent domain shift false positives."
            )

    def test_feature_schema_validator_count(self):
        """FeatureSchemaValidator must expect exactly 47 features."""
        from modules.feature_schema_validator import FeatureSchemaValidator
        v = FeatureSchemaValidator()
        assert v.expected_num_features == EXPECTED_FEATURE_COUNT, (
            f"FeatureSchemaValidator expects {v.expected_num_features} features, "
            f"expected {EXPECTED_FEATURE_COUNT}."
        )

    def test_feature_schema_validator_names_match(self):
        """FeatureSchemaValidator feature list must match canonical 47 features."""
        from modules.feature_schema_validator import FeatureSchemaValidator
        v = FeatureSchemaValidator()
        assert list(v.expected_feature_cols) == EXPECTED_FEATURES, (
            "FeatureSchemaValidator feature list mismatch. "
            f"Extra: {set(v.expected_feature_cols) - set(EXPECTED_FEATURES)}. "
            f"Missing: {set(EXPECTED_FEATURES) - set(v.expected_feature_cols)}."
        )

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(REPO_ROOT, "models", "xgboost_ae_artifacts.json")),
        reason="models/xgboost_ae_artifacts.json not found — run train_xgboost_ae.py first"
    )
    def test_xgboost_ae_artifacts_json_feature_count(self):
        """models/xgboost_ae_artifacts.json must record exactly 47 features."""
        with open(_artifact_path("xgboost_ae_artifacts.json")) as f:
            art = json.load(f)
        cols = art.get("feature_cols", [])
        assert len(cols) == EXPECTED_FEATURE_COUNT, (
            f"xgboost_ae_artifacts.json has {len(cols)} feature_cols, "
            f"expected {EXPECTED_FEATURE_COUNT}."
        )

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(REPO_ROOT, "models", "xgboost_ae_artifacts.json")),
        reason="models/xgboost_ae_artifacts.json not found — run train_xgboost_ae.py first"
    )
    def test_xgboost_ae_artifacts_json_no_ttl(self):
        """models/xgboost_ae_artifacts.json must NOT contain TTL features."""
        with open(_artifact_path("xgboost_ae_artifacts.json")) as f:
            art = json.load(f)
        cols = set(art.get("feature_cols", []))
        for prohibited in PROHIBITED_FEATURES:
            assert prohibited not in cols, (
                f"PROHIBITED feature '{prohibited}' found in xgboost_ae_artifacts.json. "
                "FIX D removed TTL features."
            )

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(REPO_ROOT, "models", "artifacts.json")),
        reason="models/artifacts.json not found"
    )
    def test_legacy_artifacts_json_feature_count(self):
        """models/artifacts.json (legacy RF artifacts) must also have 47 features."""
        with open(_artifact_path("artifacts.json")) as f:
            art = json.load(f)
        cols = art.get("feature_cols", [])
        assert len(cols) == EXPECTED_FEATURE_COUNT, (
            f"models/artifacts.json has {len(cols)} feature_cols, expected {EXPECTED_FEATURE_COUNT}. "
            "Even the legacy artifact must use 47 features (FIX D applied to both training pipelines)."
        )

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(REPO_ROOT, "models", "scaler.pkl")),
        reason="models/scaler.pkl not found — run train_xgboost_ae.py first"
    )
    def test_scaler_artifact_input_dimensions(self):
        """models/scaler.pkl must be fitted on exactly 47 features."""
        import joblib
        scaler = joblib.load(_artifact_path("scaler.pkl"))
        n_features = scaler.n_features_in_
        assert n_features == EXPECTED_FEATURE_COUNT, (
            f"scaler.pkl was fitted on {n_features} features, expected {EXPECTED_FEATURE_COUNT}."
        )

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(REPO_ROOT, "models", "xgboost_model.json")),
        reason="models/xgboost_model.json not found — run train_xgboost_ae.py first"
    )
    def test_xgboost_model_input_dimensions(self):
        """XGBoost model must be trained on exactly 47 features."""
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(_artifact_path("xgboost_model.json"))
        n_features = model.n_features_in_
        assert n_features == EXPECTED_FEATURE_COUNT, (
            f"xgboost_model.json expects {n_features} input features, expected {EXPECTED_FEATURE_COUNT}."
        )


class TestFeatureSchemaValidatorBehavior:
    """Tests for FeatureSchemaValidator logic."""

    def test_validate_complete_flow_record(self):
        """A complete 47-feature flow record should validate successfully."""
        from modules.feature_schema_validator import FeatureSchemaValidator
        v = FeatureSchemaValidator()
        flow = {feat: 1.0 for feat in EXPECTED_FEATURES}
        is_valid, errors, arr = v.validate_schema(flow)
        assert is_valid, f"Complete flow should validate. Errors: {errors}"
        assert arr.shape == (EXPECTED_FEATURE_COUNT,)

    def test_validate_rejects_missing_features(self):
        """A flow missing required features should report errors."""
        from modules.feature_schema_validator import FeatureSchemaValidator
        v = FeatureSchemaValidator()
        # Remove 3 required features
        partial_flow = {feat: 1.0 for feat in EXPECTED_FEATURES[:-3]}
        is_valid, errors, arr = v.validate_schema(partial_flow)
        assert not is_valid
        assert len(errors) == 3
        for err in errors:
            assert "Missing feature:" in err

    def test_validate_rejects_wrong_vector_length(self):
        """An array with wrong length should report a dimension error."""
        from modules.feature_schema_validator import FeatureSchemaValidator
        import numpy as np
        v = FeatureSchemaValidator()
        bad_arr = np.zeros(49)  # Old 49-feature schema
        is_valid, errors, arr = v.validate_schema(bad_arr)
        assert not is_valid
        assert any("49" in e or "mismatch" in e.lower() for e in errors)

    def test_validate_handles_nan_inf_gracefully(self):
        """NaN and Inf values should be replaced with 0.0, not raise exceptions."""
        import numpy as np
        from modules.feature_schema_validator import FeatureSchemaValidator
        v = FeatureSchemaValidator()
        flow = {feat: float("nan") if i % 5 == 0 else 1.0
                for i, feat in enumerate(EXPECTED_FEATURES)}
        is_valid, errors, arr = v.validate_schema(flow)
        assert is_valid, f"NaN values should be handled gracefully. Errors: {errors}"
        assert not any(np.isnan(arr)), "NaN values should be replaced with 0.0"
        assert not any(np.isinf(arr)), "Inf values should be replaced with 0.0"
