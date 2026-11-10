"""
scripts/experiments/run_candidate.py
====================================
Train ONE candidate pipeline (XGBoost + autoencoder + recalibrated fusion) for a feature set and evaluate it.
Everything is written to models_candidates/<run_id>/ . Production models/ is never read for writing.

Design rules enforced in code
  * three disjoint parts: TRAIN fits XGBoost / scaler / AE; CAL chooses thresholds, AE normalisation is fixed
    from TRAIN-benign data, and the fusion layer is fitted on CAL ONLY; TEST is sealed and evaluated once.
  * fit/calibrate helpers call ensure_not_sealed(), so handing them the sealed test raises.
  * individual models (XGB, AE) AND the full fusion pipeline are evaluated, with per-attack-family tables.
  * APPROVAL GATE: a full run (no --max-rows) refuses to start without --approved-full-run. Until you have
    reviewed the plan and approved, only small smoke runs (--max-rows) are possible.

Usage (smoke test only, until approval)
    python scripts/experiments/run_candidate.py --feature-set F1 --max-rows 20000 --run-id smoke-F1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config.feature_sets import FEATURE_SETS, columns, feature_matrix  # noqa: E402
from config.paths import guard_output_dir, manifest_path_for, new_candidate_dir  # noqa: E402
from evaluation.metrics import evaluate_scores, threshold_for_fpr  # noqa: E402
from evaluation.sealed import SealedTestSet, ensure_not_sealed  # noqa: E402
from evaluation.splits import chronological_split  # noqa: E402

TARGET_FPR = 0.005  # provisional acceptance target: <= 0.5 % false-positive rate per flow
TIME_COL = "FLOW_START_MILLISECONDS"


# ----------------------------------------------------------------------------------------------- utils
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


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------ fit / calibrate
def fit_base_models(train_df, feature_set, seed):
    """Fit scaler, XGBoost and the autoencoder using TRAIN data only."""
    ensure_not_sealed(train_df, context="model fitting")
    X = feature_matrix(train_df, feature_set)
    y = train_df["_y"].to_numpy()
    scaler = StandardScaler().fit(X)
    model = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8,
                              colsample_bytree=0.8, random_state=seed, n_jobs=-1, eval_metric="logloss")
    model.fit(X, y)
    Xb = scaler.transform(X[y == 0]).astype(np.float32)
    cut = int(len(Xb) * 0.8)  # chronological: earlier 80 % trains the AE, later 20 % fixes its normalisation
    enc = max(8, X.shape[1] // 2)
    ae = MLPRegressor(hidden_layer_sizes=(enc, max(4, enc // 2), enc), activation="relu", solver="adam",
                      max_iter=15, random_state=seed, early_stopping=True)
    ae.fit(Xb[:cut], Xb[:cut])
    mse_ref = np.mean((ae.predict(Xb[cut:]) - Xb[cut:]) ** 2, axis=1)
    log_ref = np.log1p(mse_ref)
    ae_stats = {"log1p_mse_mean": float(log_ref.mean()), "log1p_mse_std": float(log_ref.std() + 1e-9),
                "benign_ref_p99_mse": float(np.percentile(mse_ref, 99))}
    return scaler, model, ae, ae_stats


def score_frame(df, feature_set, scaler, model, ae, ae_stats):
    X = feature_matrix(df, feature_set)
    p_xgb = model.predict_proba(X)[:, 1]
    Xs = scaler.transform(X).astype(np.float32)
    mse = np.mean((ae.predict(Xs) - Xs) ** 2, axis=1)
    ae_enc = (np.log1p(mse) - ae_stats["log1p_mse_mean"]) / ae_stats["log1p_mse_std"]
    return {"xgb": p_xgb, "ae": ae_enc, "ae_mse": mse}


def fit_fusion_and_thresholds(cal_df, cal_scores, seed):
    """Fusion + thresholds from the CALIBRATION split only."""
    ensure_not_sealed(cal_df, context="fusion calibration")
    y = cal_df["_y"].to_numpy()
    feats = np.column_stack([_logit(cal_scores["xgb"]), cal_scores["ae"]])
    fusion = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed).fit(feats, y)
    fused = fusion.predict_proba(feats)[:, 1]
    thresholds = {
        "xgb": threshold_for_fpr(y, cal_scores["xgb"], TARGET_FPR),
        "ae": threshold_for_fpr(y, cal_scores["ae"], TARGET_FPR),
        "fusion": threshold_for_fpr(y, fused, TARGET_FPR),
    }
    return fusion, thresholds


def fused_scores(fusion, sc):
    return fusion.predict_proba(np.column_stack([_logit(sc["xgb"]), sc["ae"]]))[:, 1]


def evaluate_all(df, scores, fusion, thresholds, group_col, with_ci):
    y = df["_y"].to_numpy()
    groups = df[group_col].to_numpy()
    all_scores = {"xgb": scores["xgb"], "ae": scores["ae"], "fusion": fused_scores(fusion, scores)}
    return {name: evaluate_scores(y, all_scores[name], thresholds[name], groups=groups, with_ci=with_ci, n_boot=200)
            for name in ("xgb", "ae", "fusion")}


# ------------------------------------------------------------------------------------------------- main
def run(feature_set, run_id=None, seed=42, max_rows=None, approved_full_run=False, with_synthetic=True):
    if feature_set not in FEATURE_SETS:
        raise SystemExit(f"unknown feature set {feature_set}; choose from {sorted(FEATURE_SETS)}")
    if max_rows is None and not approved_full_run:
        raise SystemExit(
            "APPROVAL GATE: a full training run needs your explicit approval of docs/model_improvement_plan.md.\n"
            "Re-run with --approved-full-run once approved, or use --max-rows N for a smoke test."
        )
    started = time.time()
    run_id = run_id or time.strftime(f"{feature_set}-%Y%m%d-%H%M%S")
    out_dir = new_candidate_dir(run_id)  # refuses an existing non-empty candidate directory
    guard_output_dir(out_dir, targets=["xgboost_model.json", "scaler.pkl", "ae_model.pkl", "fusion_model.pkl"])
    np.random.seed(seed)

    from train_xgboost_ae import load_dataset, DATASET_PATH  # reads the dataset only
    df = load_dataset()
    if max_rows is not None and max_rows < len(df):
        df = df.iloc[np.linspace(0, len(df) - 1, int(max_rows)).astype(int)].reset_index(drop=True)
    df["_family"] = df["Attack"].astype(str).str.strip()
    df["_y"] = (df["_family"].str.lower() != "benign").astype(int)

    spec = chronological_split(df, TIME_COL, (0.6, 0.2, 0.2))
    train_df, cal_df = df.iloc[spec.train], df.iloc[spec.cal]
    sealed_test = SealedTestSet(df.iloc[spec.test], "unsw_chrono_test",
                                log_path=os.path.join(out_dir, "sealed_access.jsonl"), max_looks=1)

    scaler, model, ae, ae_stats = fit_base_models(train_df, feature_set, seed)
    cal_scores = score_frame(cal_df, feature_set, scaler, model, ae, ae_stats)
    fusion, thresholds = fit_fusion_and_thresholds(cal_df, cal_scores, seed)

    # ---- final test: sealed, evaluated exactly once -------------------------------------------------
    def _final(test_df):
        sc = score_frame(test_df, feature_set, scaler, model, ae, ae_stats)
        return evaluate_all(test_df, sc, fusion, thresholds, "_family", with_ci=True)
    test_metrics = sealed_test.evaluate(_final, purpose=f"final evaluation of {run_id}")

    results = {"unsw_chrono_test": test_metrics}
    if with_synthetic:  # provisional modern-benign / attack check (synthetic, NOT lab data)
        from evaluation.synthetic_traffic import generate
        syn = generate(seed=seed + 1000, n_sessions=3)
        syn["_y"] = syn["_label"].astype(int)
        syn_sealed = SealedTestSet(syn, "synthetic_modern", log_path=os.path.join(out_dir, "sealed_access.jsonl"))
        results["synthetic_modern_PROVISIONAL"] = syn_sealed.evaluate(
            lambda d: evaluate_all(d.assign(_group=d["_traffic_class"]),
                                   score_frame(d, feature_set, scaler, model, ae, ae_stats),
                                   fusion, thresholds, "_group", with_ci=False),
            purpose="provisional synthetic evaluation")

    # ---- persist (candidate directory only) ---------------------------------------------------------------
    model.save_model(os.path.join(out_dir, "xgboost_model.json"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(ae, os.path.join(out_dir, "ae_model.pkl"))
    joblib.dump(fusion, os.path.join(out_dir, "fusion_model.pkl"))
    with open(os.path.join(out_dir, "candidate_artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable({
            "feature_set": feature_set, "feature_columns": columns(feature_set), "thresholds": thresholds,
            "target_fpr": TARGET_FPR, "ae_normalisation": ae_stats,
            "fusion": {"inputs": ["logit(xgb_prob)", "ae_standardised_log1p_mse"],
                       "coef": fusion.coef_[0].tolist(), "intercept": float(fusion.intercept_[0])},
        }), f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2)
    from modules.trusted_model_loader import regenerate_model_manifest
    regenerate_model_manifest(out_dir, manifest_path=manifest_path_for(out_dir))
    manifest = {
        "run_id": run_id, "feature_set": feature_set, "seed": seed, "smoke_max_rows": max_rows,
        "approved_full_run": bool(approved_full_run), "git_commit": _git_commit(),
        "dataset": {"path": os.path.basename(DATASET_PATH), "sha256": _sha256(DATASET_PATH), "rows_used": len(df)},
        "split": spec.manifest(), "sealed_looks": sealed_test.looks + (syn_sealed.looks if with_synthetic else []),
        "versions": {m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "xgboost")},
        "duration_s": round(time.time() - started, 1),
        "synthetic_note": "synthetic_modern_PROVISIONAL uses generated traffic, not lab captures",
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2)
    print(f"[run_candidate] {run_id}: wrote {out_dir}")
    return out_dir, results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SETS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke test: evenly spaced subsample")
    ap.add_argument("--approved-full-run", action="store_true", help="ONLY after you approved the plan")
    ap.add_argument("--no-synthetic", action="store_true")
    a = ap.parse_args(argv)
    run(a.feature_set, a.run_id, a.seed, a.max_rows, a.approved_full_run, with_synthetic=not a.no_synthetic)


if __name__ == "__main__":
    main()
