"""
modules/feature_schema_validator.py
====================================
LIVE TRAFFIC FEATURE SCHEMA VALIDATOR

Validates that live packet flow capture engines produce EXACTLY the same 47 NetFlow/IPFIX
feature columns in the exact order expected by the trained XGBoost, Autoencoder, and GNN models.
(FIX D: MIN_TTL and MAX_TTL removed — 47 features instead of the original 49)

Schema Definition:
  - 47 Features matching NF-UNSW-NB15-v3 (CANONICAL — post FIX D TTL elimination)
    Original design was 49 features; MIN_TTL and MAX_TTL were removed because they
    caused ~99.7% false positives on real traffic (TTL 64/128 vs dataset-synthetic 31/254).
  - 11 Core Operational Fields for SOC Dashboard Display:
      [Source HOST, Destination HOST, Service/Port, Protocol, Bytes, Packets, Duration, Model Risk, Attack Path, XAI Explanation, Level]

"""

import os
import json
import numpy as np
import pandas as pd
from config.risk_thresholds import RISK_L1_MAX, RISK_L2_MAX
from train_xgboost_ae import FEATURE_COLS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")
ARTIFACTS_PATH = os.path.join(MODELS_DIR, "xgboost_ae_artifacts.json")


class FeatureSchemaValidator:
    """Validates Live Traffic Capture Feature Schema against Offline Model Schema."""

    def __init__(self):
        self.expected_feature_cols = FEATURE_COLS
        self.expected_num_features = len(FEATURE_COLS)

    def validate_schema(self, live_flow_record):
        """
        Validates a live flow record dictionary or feature vector.
        Returns (is_valid: bool, errors: list, validated_vector: np.ndarray)
        """
        errors = []

        if isinstance(live_flow_record, dict):
            # Verify feature presence and order
            feat_vals = []
            for col in self.expected_feature_cols:
                if col not in live_flow_record:
                    # FIX C: Record missing-feature error instead of silently defaulting.
                    # The 0.0 default is still appended so the vector length is correct,
                    # but the error is now tracked for strict validation enforcement.
                    errors.append(f"Missing feature: {col}")
                    feat_vals.append(0.0)
                else:
                    try:
                        val = float(live_flow_record[col])
                        if np.isnan(val) or np.isinf(val):
                            val = 0.0
                        feat_vals.append(val)
                    except (ValueError, TypeError):
                        feat_vals.append(0.0)

            feat_arr = np.array(feat_vals, dtype=np.float32)

        elif isinstance(live_flow_record, (list, np.ndarray)):
            feat_arr = np.array(live_flow_record, dtype=np.float32)
            if len(feat_arr) != self.expected_num_features:
                errors.append(
                    f"Feature length mismatch: Expected {self.expected_num_features}, got {len(feat_arr)}."
                )
        else:
            errors.append(f"Invalid live_flow_record type: {type(live_flow_record)}.")
            return False, errors, None

        if len(feat_arr) != self.expected_num_features:
            errors.append(
                f"Feature vector length {len(feat_arr)} does not match expected {self.expected_num_features}."
            )

        is_valid = len(errors) == 0
        return is_valid, errors, feat_arr

    def extract_operational_fields(self, live_flow_dict, model_risk=0.0, attack_path=None, xai_exp=None):
        """
        Extracts the 11 core operational fields required for SOC Dashboard display.
        """
        # FIX B: Prioritize _src_ip/_dst_ip from FlowExtractor metadata to avoid
        # port-as-host misrouting in the Bipartite Graph. Fall back to IPV4_SRC_ADDR
        # for offline dataset compatibility (dataset rows have IPV4_SRC_ADDR but no _src_ip).
        src_ip = str(live_flow_dict.get("_src_ip", live_flow_dict.get("IPV4_SRC_ADDR", "192.168.1.10")))
        dst_ip = str(live_flow_dict.get("_dst_ip", live_flow_dict.get("IPV4_DST_ADDR", "10.40.182.1")))
        dst_port = int(live_flow_dict.get("_dst_port", live_flow_dict.get("L4_DST_PORT", 80)))
        protocol = int(live_flow_dict.get("PROTOCOL", 6))
        byte_cnt = float(live_flow_dict.get("IN_BYTES", 100.0))
        pkt_cnt = float(live_flow_dict.get("IN_PKTS", 2.0))
        duration = float(live_flow_dict.get("FLOW_DURATION_MILLISECONDS", 1.0))

        level = "Level 1 — Normal"
        if model_risk >= RISK_L2_MAX:
            level = "Level 3 — High Risk"
        elif model_risk >= RISK_L1_MAX:
            level = "Level 2 — Suspicious"

        return {
            "source_host": f"HOST:{src_ip}",
            "destination_host": f"HOST:{dst_ip}",
            "service_port": f"SERVICE:{dst_port}/{protocol}",
            "protocol": protocol,
            "bytes": byte_cnt,
            "packets": pkt_cnt,
            "duration": duration,
            "model_risk": float(model_risk),
            "attack_path": attack_path or [f"HOST:{src_ip}", f"SERVICE:{dst_port}/{protocol}", f"HOST:{dst_ip}"],
            "xai_explanation": xai_exp or {},
            "level": level,
        }


if __name__ == "__main__":
    validator = FeatureSchemaValidator()
    print(f"Feature Schema Validator initialized.")
    print(f"Expected Feature Count: {validator.expected_num_features}")
    print(f"Feature Schema Sample: {validator.expected_feature_cols[:5]} ...")
