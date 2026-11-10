"""
scripts/experiments/ae_historical_vs_modern.py
==============================================
How does the autoencoder behave on HISTORICAL benign traffic (UNSW-NB15, 2015) versus MODERN benign traffic?

Inference only. Nothing is trained or written under models/.
Modern benign comes from:
  * synthetic traffic through the real FlowExtractor (DNS, HTTPS, QUIC, NTP, DHCP, mDNS, SSDP, ICMP, IGMP, TCP/UDP)
    -> PROVISIONAL, it is a model of traffic, not traffic;
  * the earlier ad-hoc live capture in outputs/fp_live_flows.json, if present (presumed benign, not a lab capture).

Default model: the production autoencoder + scaler (feature set F0). Use --candidate <dir> to evaluate a candidate
produced by run_candidate.py (its own feature set, scaler, AE and normalisation).

    python scripts/experiments/ae_historical_vs_modern.py
    python scripts/experiments/ae_historical_vs_modern.py --candidate models_candidates/<run_id>
"""
import argparse
import json
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config.feature_sets import feature_matrix  # noqa: E402
from evaluation.metrics import roc_auc  # noqa: E402
from evaluation.synthetic_traffic import generate  # noqa: E402
from modules.trusted_model_loader import safe_load_joblib  # noqa: E402
from train_xgboost_ae import FEATURE_COLS  # noqa: E402


def load_model(candidate):
    if candidate is None:
        art = json.load(open(os.path.join(ROOT, "models", "xgboost_ae_artifacts.json")))
        d = os.path.join(ROOT, "models")
        return {"set": "F0", "scaler": safe_load_joblib(os.path.join(d, "scaler.pkl")),
                "ae": safe_load_joblib(os.path.join(d, "ae_model.pkl")), "threshold_mse": float(art["ae_threshold"]),
                "encode": lambda mse: mse, "name": "production"}
    d = os.path.abspath(candidate)
    art = json.load(open(os.path.join(d, "candidate_artifacts.json")))
    st = art["ae_normalisation"]
    enc = lambda mse: (np.log1p(mse) - st["log1p_mse_mean"]) / st["log1p_mse_std"]  # noqa: E731
    return {"set": art["feature_set"], "scaler": safe_load_joblib(os.path.join(d, "scaler.pkl")),
            "ae": safe_load_joblib(os.path.join(d, "ae_model.pkl")), "threshold_mse": art["thresholds"]["ae"],
            "encode": enc, "name": os.path.basename(d)}


def ae_score(model, df):
    X = feature_matrix(df, model["set"])
    Xs = model["scaler"].transform(X).astype(np.float32)
    mse = np.mean((model["ae"].predict(Xs) - Xs) ** 2, axis=1)
    return model["encode"](mse)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "ae_historical_vs_modern.json"))
    a = ap.parse_args(argv)
    model = load_model(a.candidate)
    thr = model["threshold_mse"]

    uns = pd.read_csv(os.path.join(ROOT, "NF-UNSW-NB15-v3.csv"), usecols=FEATURE_COLS + ["Attack", "FLOW_START_MILLISECONDS"])
    uns = uns.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    uns = uns.iloc[int(len(uns) * 0.75):]  # the chronologically latest 25 % (never seen in the original training)
    uy = (uns["Attack"].astype(str).str.strip().str.lower() != "benign").astype(int).to_numpy()
    s_hist = ae_score(model, uns)

    syn = generate(seed=a.seed, n_sessions=3)
    sy = syn["_label"].astype(int).to_numpy()
    s_syn = ae_score(model, syn)

    rows = []

    def add(name, scores, label_note):
        rows.append({"domain": name, "note": label_note, "n": int(len(scores)),
                     "median_score": float(np.median(scores)), "p99_score": float(np.quantile(scores, .99)),
                     "share_above_threshold": float((scores > thr).mean())})

    add("HISTORICAL benign (UNSW 2015, latest 25%)", s_hist[uy == 0], "real, 2015 testbed")
    for cls in sorted(syn.loc[syn._label == 0, "_traffic_class"].unique()):
        add(f"MODERN synthetic benign: {cls}", s_syn[(sy == 0) & (syn._traffic_class == cls).to_numpy()], "synthetic")
    add("MODERN synthetic benign: ALL", s_syn[sy == 0], "synthetic")
    live_path = os.path.join(ROOT, "outputs", "fp_live_flows.json")
    if os.path.exists(live_path):
        live = pd.DataFrame([dict(zip(FEATURE_COLS, r["vec"])) for r in json.load(open(live_path))])
        add("MODERN real ad-hoc capture (presumed benign)", ae_score(model, live), "real, not a lab capture, unlabelled")

    within = {
        "historical_UNSW_benign_vs_attack_ROC_AUC": roc_auc(uy, s_hist),
        "modern_synthetic_benign_vs_attack_ROC_AUC (PROVISIONAL)": roc_auc(sy, s_syn),
    }
    out = {"model": model["name"], "feature_set": model["set"], "threshold": thr,
           "threshold_unit": "mse (production)" if a.candidate is None else "standardised log1p(mse)",
           "rows": rows, "ranking_quality": within,
           "caveat": "Synthetic traffic is a model of traffic. Treat modern-synthetic rows as provisional."}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"AE model: {model['name']} ({model['set']}) | anomaly threshold = {thr:.4f} ({out['threshold_unit']})")
    print(f"{'domain':58s}{'n':>7s}{'median':>10s}{'p99':>11s}{'>thr':>8s}")
    for r in rows:
        print(f"{r['domain'][:57]:58s}{r['n']:7d}{r['median_score']:10.4f}{r['p99_score']:11.3f}{100 * r['share_above_threshold']:7.1f}%")
    print("\nranking quality (can the AE still separate attacks from benign INSIDE each domain?):")
    for k, v in within.items():
        print(f"  {k}: {v:.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
