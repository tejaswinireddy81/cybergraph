"""
scripts/experiments/evaluate_production_reference.py
====================================================
Evaluates the CURRENT PRODUCTION models (F0) as the reference baseline across:
  1. The chronological UNSW test split (latest 20%, never seen in train/cal).
  2. The modern synthetic traffic benchmark (DNS, HTTPS, QUIC, TCP, UDP, ICMP, IGMP).

Inference only. Preserves all production models, does not touch live firewall or retrain anything.
Outputs:
  outputs/production_reference_metrics.json
"""
import json
import math
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from config.feature_sets import columns, feature_matrix  # noqa: E402
from evaluation.metrics import evaluate_scores  # noqa: E402
from evaluation.sealed import SealedTestSet  # noqa: E402
from evaluation.splits import chronological_split  # noqa: E402
from evaluation.synthetic_traffic import generate  # noqa: E402
from modules.trusted_model_loader import safe_load_joblib  # noqa: E402

DATASET_PATH = os.path.join(ROOT, "NF-UNSW-NB15-v3.csv")
MODELS_DIR = os.path.join(ROOT, "models")
TIME_COL = "FLOW_START_MILLISECONDS"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(float(o)) or math.isinf(float(o))) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def load_production_pipeline():
    artifacts = json.load(open(os.path.join(MODELS_DIR, "xgboost_ae_artifacts.json"), "r"))
    scaler = safe_load_joblib(os.path.join(MODELS_DIR, "scaler.pkl"))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
    ae_model = safe_load_joblib(os.path.join(MODELS_DIR, "ae_model.pkl"))
    fusion_model = safe_load_joblib(os.path.join(MODELS_DIR, "fusion_model.pkl"))

    return {
        "feature_set": "F0",
        "feature_cols": columns("F0"),
        "artifacts": artifacts,
        "scaler": scaler,
        "xgb": xgb_model,
        "ae": ae_model,
        "fusion": fusion_model,
        "thresholds": {
            "xgb": 0.75,  # Dashboard Critical / Alert Threshold
            "ae": float(artifacts.get("ae_threshold", 0.075)),
            "fusion": 0.50,
        },
    }


def score_dataframe(pipe, df):
    X = feature_matrix(df, pipe["feature_set"])
    feats = pipe["feature_cols"]
    X_aligned = X[feats]
    X_scaled = pipe["scaler"].transform(X_aligned).astype(np.float32)

    # XGBoost score
    xgb_prob = pipe["xgb"].predict_proba(X_aligned)[:, 1]

    # AE reconstruction error
    ae_pred = pipe["ae"].predict(X_scaled)
    ae_mse = np.mean((ae_pred - X_scaled) ** 2, axis=1)

    # Historical fusion normalization: min(mse / ae_mse_max, 1.0)
    ae_max = float(pipe["artifacts"].get("ae_mse_max", 51.691))
    ae_norm = np.clip(ae_mse / max(ae_max, 1e-6), 0.0, 1.0)

    # Fused input
    X_fus = np.column_stack([xgb_prob, ae_norm])
    fus_prob = pipe["fusion"].predict_proba(X_fus)[:, 1]

    return {
        "xgb_score": xgb_prob,
        "ae_mse": ae_mse,
        "ae_norm": ae_norm,
        "fusion_score": fus_prob,
    }


def main():
    print("======================================================================")
    print("CYBERGRAPH - PRODUCTION MODEL REFERENCE BENCHMARK (F0)")
    print("======================================================================")
    pipe = load_production_pipeline()

    # 1. Evaluate on UNSW Chronological Test Split
    print(f"\n[1/2] Loading UNSW dataset from {DATASET_PATH} ...")
    cols_to_load = pipe["feature_cols"] + ["Label", "Attack", TIME_COL]
    df_raw = pd.read_csv(DATASET_PATH, usecols=cols_to_load)
    df_raw = df_raw.sort_values(TIME_COL).reset_index(drop=True)

    spec = chronological_split(df_raw, TIME_COL, fractions=(0.6, 0.2, 0.2))
    test_df = df_raw.iloc[spec.test].copy()
    y_test = (test_df["Attack"].astype(str).str.strip().str.lower() != "benign").astype(int).to_numpy()
    test_df["_y"] = y_test
    test_df["_family"] = test_df["Attack"].astype(str).str.strip()

    scores_unsw = score_dataframe(pipe, test_df)
    unsw_results = {
        "xgboost": evaluate_scores(y_test, scores_unsw["xgb_score"], threshold=pipe["thresholds"]["xgb"],
                                   groups=test_df["_family"].to_numpy()),
        "autoencoder": evaluate_scores(y_test, scores_unsw["ae_mse"], threshold=pipe["thresholds"]["ae"],
                                       groups=test_df["_family"].to_numpy()),
        "fusion": evaluate_scores(y_test, scores_unsw["fusion_score"], threshold=pipe["thresholds"]["fusion"],
                                  groups=test_df["_family"].to_numpy()),
    }

    # 2. Evaluate on Modern Synthetic Benchmark
    print("[2/2] Generating modern synthetic traffic benchmark ...")
    syn_df = generate(seed=42, n_sessions=3)
    y_syn = syn_df["_label"].astype(int).to_numpy()
    syn_df["_y"] = y_syn

    scores_syn = score_dataframe(pipe, syn_df)
    syn_results = {
        "xgboost": evaluate_scores(y_syn, scores_syn["xgb_score"], threshold=pipe["thresholds"]["xgb"],
                                   groups=syn_df["_traffic_class"].to_numpy()),
        "autoencoder": evaluate_scores(y_syn, scores_syn["ae_mse"], threshold=pipe["thresholds"]["ae"],
                                       groups=syn_df["_traffic_class"].to_numpy()),
        "fusion": evaluate_scores(y_syn, scores_syn["fusion_score"], threshold=pipe["thresholds"]["fusion"],
                                  groups=syn_df["_traffic_class"].to_numpy()),
    }

    report = {
        "model_set": "F0_production",
        "thresholds": pipe["thresholds"],
        "unsw_chronological_test": unsw_results,
        "modern_synthetic_provisional": syn_results,
    }

    out_file = os.path.join(ROOT, "outputs", "production_reference_metrics.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 70)
    print("PRODUCTION (F0) REFERENCE RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Domain / Test Set':<32} | {'Model':<9} | {'ROC-AUC':<7} | {'Recall':<7} | {'FPR':<7} | {'F1':<7}")
    print("-" * 70)
    for dom_name, dom_data in [("UNSW Chrono Test (20%)", unsw_results), ("Modern Synthetic (Provisional)", syn_results)]:
        for mname in ("xgboost", "autoencoder", "fusion"):
            m = dom_data[mname]
            at_t = m.get("at_threshold", {})
            auc_str = f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") is not None else "N/A"
            rec_str = f"{100 * at_t.get('recall', 0):5.1f}%" if at_t.get("recall") is not None else "N/A"
            fpr_str = f"{100 * at_t.get('fpr', 0):5.1f}%" if at_t.get("fpr") is not None else "N/A"
            f1_str = f"{at_t.get('f1', 0):.4f}" if at_t.get("f1") is not None else "N/A"
            print(f"{dom_name:<32} | {mname:<9} | {auc_str:<7} | {rec_str:<7} | {fpr_str:<7} | {f1_str:<7}")

    print("\nPer-Attack Family Recall on UNSW Chrono Test (XGBoost @ 0.75):")
    fam_rows = unsw_results["xgboost"].get("per_group", [])
    for row in fam_rows:
        fam = row.get("group", "")
        if fam.lower() != "benign":
            rec = row.get("recall", 0.0)
            n_tot = row.get("n", 0)
            print(f"  - {fam:<16}: {100 * rec:5.1f}% (n={n_tot})")

    print("\nPer-Class FPR on Modern Synthetic Benign (XGBoost @ 0.75):")
    cls_rows = syn_results["xgboost"].get("per_group", [])
    for row in cls_rows:
        cls = row.get("group", "")
        fpr = row.get("fpr", 0.0)
        n_tot = row.get("n_benign", 0)
        if n_tot > 0:
            print(f"  - {cls:<16}: {100 * fpr:5.1f}% (n={n_tot})")

    print(f"\n[DONE] Saved complete reference metrics to: {out_file}\n")


if __name__ == "__main__":
    main()
