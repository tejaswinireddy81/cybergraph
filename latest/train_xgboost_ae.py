"""
train_xgboost_ae.py
===================
PHASE 1: XGBoost + Autoencoder + Fusion Engine (Temporal Chronological Splitting)

Reads real NF-UNSW-NB15-v3.csv dataset, applies strict TEMPORAL CHRONOLOGICAL SPLITTING:
  - Train set: Earliest 75% of flow timeline
  - Test set: Final 25% of flow timeline (future unseen traffic)

Trains:
  1. XGBoost Classifier (on flow features)
  2. Neural Autoencoder Anomaly Detector (MLPRegressor MSE reconstruction on normal flows)
  3. Logistic Regression Fusion Layer (combines XGBoost prob + AE normalized MSE)

Saves artifacts to models/:
  - models/xgboost_model.json
  - models/ae_model.pkl
  - models/scaler.pkl
  - models/fusion_model.pkl
  - models/xgboost_ae_artifacts.json
"""

import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import xgboost as xgb

from config.paths import resolve_models_dir, guard_output_dir, manifest_path_for

warnings.filterwarnings("ignore")

# --- Reproducibility ---
random.seed(42)
np.random.seed(42)

# --- Paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "NF-UNSW-NB15-v3.csv")
MODELS_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

XGB_PATH = os.path.join(MODELS_DIR, "xgboost_model.json")
AE_PATH = os.path.join(MODELS_DIR, "ae_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FUSION_PATH = os.path.join(MODELS_DIR, "fusion_model.pkl")
ARTIFACTS_PATH = os.path.join(MODELS_DIR, "xgboost_ae_artifacts.json")

# --- 47 Feature Columns from NF-UNSW-NB15-v3 ---
# FIX D (TTL DOMAIN SHIFT): MIN_TTL and MAX_TTL have been REMOVED.
# The NF-UNSW-NB15-v3 dataset has a severe TTL bias (benign=31/32, attack=254)
# that causes ~99.7% false positives on real-world Wi-Fi traffic (TTL 64/128).
# XGBoost relied 88.4% on these two features (MIN_TTL=61.25%, MAX_TTL=27.14%).
# Removing them forces the models to learn genuinely discriminative patterns.
FEATURE_COLS = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    # FIX D: MIN_TTL and MAX_TTL REMOVED — see comment above
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
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
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"DATASET MISSING ERROR: Real dataset not found at '{DATASET_PATH}'. "
            "Please place 'NF-UNSW-NB15-v3.csv' in the project root directory to proceed."
        )
    print(f"Loading offline dataset from: {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    if "Attack" in df.columns:
        df["label_enc"] = np.where(
            df["Attack"].astype(str).str.strip().str.lower() == "benign", 0, 1
        ).astype(np.int8)
    elif "Label" in df.columns:
        df["label_enc"] = df["Label"].astype(np.int8)
    else:
        raise KeyError("Neither 'Attack' nor 'Label' column found in dataset.")

    # Sort strictly chronologically by FLOW_START_MILLISECONDS
    if "FLOW_START_MILLISECONDS" in df.columns:
        df = df.sort_values(by="FLOW_START_MILLISECONDS").reset_index(drop=True)

    return df


def train_pipeline(models_dir=None, overwrite_production=False, max_rows=None, seed=42):
    """Train XGBoost + autoencoder + fusion.

    models_dir : output directory. Default (None) = CYBERGRAPH_MODELS_DIR or the production models/ dir.
                 Candidate runs should use models_candidates/<run_id>/.
    overwrite_production : needed to overwrite existing production models (a backup is taken first).
    max_rows : evenly-spaced chronological subsample, for smoke tests only (default: use everything).
    seed : random seed (default 42 = historical behaviour).
    """
    models_dir = resolve_models_dir(models_dir)
    guard_info = guard_output_dir(
        models_dir, overwrite_production=overwrite_production,
        targets=["xgboost_model.json", "ae_model.pkl", "scaler.pkl", "fusion_model.pkl", "xgboost_ae_artifacts.json"],
    )
    if guard_info["backup"]:
        print(f"[BACKUP] Existing production models copied to {guard_info['backup']}")
    xgb_path = os.path.join(models_dir, "xgboost_model.json")
    ae_path = os.path.join(models_dir, "ae_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    fusion_path = os.path.join(models_dir, "fusion_model.pkl")
    artifacts_path = os.path.join(models_dir, "xgboost_ae_artifacts.json")
    print(f"[OUTPUT] models will be written to: {models_dir}")
    random.seed(seed)
    np.random.seed(seed)

    df = load_dataset()
    if max_rows is not None and max_rows < len(df):
        df = df.iloc[np.linspace(0, len(df) - 1, int(max_rows)).astype(int)].reset_index(drop=True)
        print(f"[SMOKE] using an evenly spaced subsample of {len(df)} rows")

    X = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = df["label_enc"]

    # Strict Chronological Train (first 75%) / Test (final 25%) Split
    train_size = int(len(df) * 0.75)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    print(f"Chronological Train split: {X_train.shape[0]} samples (First 75% of timeline)")
    print(f"Chronological Test split : {X_test.shape[0]} samples (Final 25% of timeline)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # ============================================================
    # 1. XGBOOST CLASSIFIER
    # ============================================================
    print("\n[Phase 1.1] Training XGBoost Classifier ...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_prec = precision_score(y_test, xgb_preds, zero_division=0)
    xgb_rec = recall_score(y_test, xgb_preds, zero_division=0)
    xgb_f1 = f1_score(y_test, xgb_preds, zero_division=0)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    xgb_cm = confusion_matrix(y_test, xgb_preds).tolist()

    print(f"XGBoost Test Accuracy : {xgb_acc:.6f}")
    print(f"XGBoost Test Precision: {xgb_prec:.6f}")
    print(f"XGBoost Test Recall   : {xgb_rec:.6f}")
    print(f"XGBoost Test F1 Score : {xgb_f1:.6f}")
    print(f"XGBoost Test ROC-AUC  : {xgb_auc:.6f}")
    print(f"XGBoost Confusion Matrix: {xgb_cm}")

    # ============================================================
    # 2. AUTOENCODER ANOMALY DETECTOR
    # ============================================================
    print("\n[Phase 1.2] Training Neural Autoencoder Anomaly Detector on normal flows ...")
    normal_mask = y_train.values == 0
    X_normal_s = X_train_s[normal_mask]

    # Chronological split for val set inside normal training flows
    normal_train_size = int(len(X_normal_s) * 0.80)
    Xn_train_s = X_normal_s[:normal_train_size]
    Xn_val_s = X_normal_s[normal_train_size:]

    input_dim = Xn_train_s.shape[1]
    encoding_dim = max(8, input_dim // 2)

    ae_model = MLPRegressor(
        hidden_layer_sizes=(encoding_dim, max(4, encoding_dim // 2), encoding_dim),
        activation="relu",
        solver="adam",
        max_iter=15,
        random_state=seed,
        early_stopping=True,
    )
    ae_model.fit(Xn_train_s, Xn_train_s)

    recon_val = ae_model.predict(Xn_val_s)
    mse_val = np.mean(np.square(recon_val - Xn_val_s), axis=1)
    ae_threshold = float(np.percentile(mse_val, 99))

    recon_test = ae_model.predict(X_test_s)
    mse_test = np.mean(np.square(recon_test - X_test_s), axis=1)
    ae_preds = (mse_test > ae_threshold).astype(np.int8)
    ae_mse_max = float(np.percentile(mse_test, 99.9))

    ae_acc = accuracy_score(y_test, ae_preds)
    ae_prec = precision_score(y_test, ae_preds, zero_division=0)
    ae_rec = recall_score(y_test, ae_preds, zero_division=0)
    ae_f1 = f1_score(y_test, ae_preds, zero_division=0)
    ae_cm = confusion_matrix(y_test, ae_preds).tolist()

    print(f"Autoencoder Threshold : {ae_threshold:.8f}")
    print(f"Autoencoder Test Acc  : {ae_acc:.6f}")
    print(f"Autoencoder Test F1   : {ae_f1:.6f}")
    print(f"Autoencoder Confusion Matrix: {ae_cm}")

    # ============================================================
    # 3. FUSION ENGINE (Logistic Regression Stacking)
    # ============================================================
    print("\n[Phase 1.3] Training Fusion Engine ...")
    ae_norm = np.minimum(mse_test / (ae_mse_max + 1e-9), 1.0)
    fusion_X = np.column_stack([xgb_probs, ae_norm])
    fusion_y = y_test.values

    fusion_train_size = int(len(fusion_X) * 0.70)
    Xf_train, Xf_test = fusion_X[:fusion_train_size], fusion_X[fusion_train_size:]
    yf_train, yf_test = fusion_y[:fusion_train_size], fusion_y[fusion_train_size:]

    fusion_model = LogisticRegression(class_weight="balanced", max_iter=1000)
    fusion_model.fit(Xf_train, yf_train)

    fusion_preds = fusion_model.predict(Xf_test)
    fusion_probs = fusion_model.predict_proba(Xf_test)[:, 1]

    fusion_acc = accuracy_score(yf_test, fusion_preds)
    fusion_prec = precision_score(yf_test, fusion_preds, zero_division=0)
    fusion_rec = recall_score(yf_test, fusion_preds, zero_division=0)
    fusion_f1 = f1_score(yf_test, fusion_preds, zero_division=0)
    fusion_auc = roc_auc_score(yf_test, fusion_probs)
    fusion_cm = confusion_matrix(yf_test, fusion_preds).tolist()

    print(f"Fusion Test Accuracy : {fusion_acc:.6f}")
    print(f"Fusion Test F1 Score : {fusion_f1:.6f}")
    print(f"Fusion Test ROC-AUC  : {fusion_auc:.6f}")
    print(f"Fusion Confusion Matrix: {fusion_cm}")

    # ============================================================
    # 4. SAVE ARTIFACTS
    # ============================================================
    print("\nSaving Phase 1 trained model artifacts ...")
    xgb_model.save_model(xgb_path)
    joblib.dump(ae_model, ae_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(fusion_model, fusion_path)

    artifacts = {
        "feature_cols": FEATURE_COLS,
        "ae_threshold": ae_threshold,
        "ae_mse_max": ae_mse_max,
        "metrics": {
            "xgboost": {
                "accuracy": float(xgb_acc),
                "precision": float(xgb_prec),
                "recall": float(xgb_rec),
                "f1": float(xgb_f1),
                "auc": float(xgb_auc),
                "confusion_matrix": xgb_cm,
            },
            "autoencoder": {
                "threshold": float(ae_threshold),
                "accuracy": float(ae_acc),
                "precision": float(ae_prec),
                "recall": float(ae_rec),
                "f1": float(ae_f1),
                "confusion_matrix": ae_cm,
            },
            "fusion": {
                "accuracy": float(fusion_acc),
                "precision": float(fusion_prec),
                "recall": float(fusion_rec),
                "f1": float(fusion_f1),
                "auc": float(fusion_auc),
                "confusion_matrix": fusion_cm,
            },
        },
    }

    with open(artifacts_path, "w") as f:
        json.dump(artifacts, f, indent=2)

    try:
        from modules.trusted_model_loader import regenerate_model_manifest
        regenerate_model_manifest(models_dir, manifest_path=manifest_path_for(models_dir))
    except Exception as e:
        print(f"[WARN] Unable to regenerate manifest: {e}")

    print(f"Phase 1 artifacts saved successfully to '{models_dir}'!")
    return artifacts


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Train XGBoost + autoencoder + fusion (writes model files).")
    ap.add_argument("--models-dir", default=None,
                    help="output directory (default: $CYBERGRAPH_MODELS_DIR or production models/). "
                         "Use models_candidates/<run_id> for candidate runs.")
    ap.add_argument("--overwrite-production", action="store_true",
                    help="allow replacing existing production models (a backup is taken first)")
    ap.add_argument("--max-rows", type=int, default=None, help="evenly spaced subsample, smoke tests only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    train_pipeline(models_dir=args.models_dir, overwrite_production=args.overwrite_production,
                   max_rows=args.max_rows, seed=args.seed)


if __name__ == "__main__":
    _main()
