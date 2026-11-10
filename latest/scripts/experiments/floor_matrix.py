"""
scripts/experiments/floor_matrix.py
===================================
TCP 52-byte floor ON vs OFF, for each model/feature set. Inference only; nothing is trained or overwritten.

For every model it generates the SAME synthetic sessions twice (floor=52 and floor=0), scores both, and reports
  * whether the floor changes the model's input at all (it cannot for sets that no longer contain
    MIN_IP_PKT_LEN / SHORTEST_FLOW_PKT),
  * false-positive rate on benign traffic (overall, TCP, and the one-packet "orphan" TCP flows the extractor
    creates after the first FIN/RST),
  * recall per attack family.

Models evaluated: the production F0 model (alert threshold 0.75 = dashboard "critical"), plus any candidates passed
with --candidate (each uses its own feature set and its calibrated XGBoost threshold).

Synthetic traffic is PROVISIONAL (see evaluation/synthetic_traffic.py). Lab captures replace it once approved.

    python scripts/experiments/floor_matrix.py
    python scripts/experiments/floor_matrix.py --candidate models_candidates/F1-... --candidate models_candidates/F2-...
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
import xgboost as xgb  # noqa: E402

from config.feature_sets import feature_matrix  # noqa: E402
from evaluation.synthetic_traffic import generate  # noqa: E402


def load_models(candidates):
    prod = xgb.XGBClassifier()
    prod.load_model(os.path.join(ROOT, "models", "xgboost_model.json"))
    out = [{"name": "production (F0)", "set": "F0", "model": prod, "threshold": 0.75}]
    for c in candidates:
        d = os.path.abspath(c)
        art = json.load(open(os.path.join(d, "candidate_artifacts.json")))
        m = xgb.XGBClassifier()
        m.load_model(os.path.join(d, "xgboost_model.json"))
        out.append({"name": f"{os.path.basename(d)} ({art['feature_set']})", "set": art["feature_set"], "model": m,
                    "threshold": art["thresholds"]["xgb"]})
    return out


def summarise(df, X, entry):
    p = entry["model"].predict_proba(X)[:, 1]
    alert = p >= entry["threshold"]
    y = df["_label"].to_numpy().astype(int)
    tcp = (df["PROTOCOL"] == 6).to_numpy()
    orphan = tcp & ((df["IN_PKTS"] + df["OUT_PKTS"]) <= 1).to_numpy() & (y == 0)
    res = {
        "benign_FPR": float(alert[y == 0].mean()),
        "benign_TCP_FPR": float(alert[(y == 0) & tcp].mean()),
        "orphan_TCP_FPR": float(alert[orphan].mean()) if orphan.any() else None,
        "n_orphans": int(orphan.sum()), "n_benign": int((y == 0).sum()),
    }
    for fam in sorted(set(df.loc[y == 1, "_family"])):
        m = (df["_family"] == fam).to_numpy()
        res[f"recall_{fam}"] = float(alert[m].mean())
    res["recall_ALL_attacks"] = float(alert[y == 1].mean())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", default=[])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "floor_matrix.json"))
    a = ap.parse_args(argv)
    models = load_models(a.candidate)
    on = generate(seed=a.seed, n_sessions=3, floor=52)
    off = generate(seed=a.seed, n_sessions=3, floor=0)
    assert on.shape == off.shape
    results = {}
    print(f"synthetic sessions: {len(on)} flows (same seed for floor ON and OFF) | PROVISIONAL, not lab data\n")
    for e in models:
        Xon, Xoff = feature_matrix(on, e["set"]), feature_matrix(off, e["set"])
        same = bool(np.array_equal(Xon.to_numpy(), Xoff.to_numpy()))
        r_on, r_off = summarise(on, Xon, e), summarise(off, Xoff, e)
        results[e["name"]] = {"floor_changes_model_input": not same, "floor_ON": r_on, "floor_OFF": r_off}
        print(f"=== {e['name']}   threshold={e['threshold']:.4f}   floor changes the model input: {'YES' if not same else 'NO (floor is dead code for this set)'}")
        keys = ["benign_FPR", "benign_TCP_FPR", "orphan_TCP_FPR"] + [k for k in r_on if k.startswith("recall_")]
        print(f"    {'metric':26s}{'floor ON':>10s}{'floor OFF':>11s}")
        for k in keys:
            fmt = lambda v: "   n/a" if v is None else f"{100 * v:8.1f}%"  # noqa: E731
            print(f"    {k:26s}{fmt(r_on[k]):>10s}{fmt(r_off[k]):>11s}")
        print(f"    (benign flows={r_on['n_benign']}, one-packet TCP orphans={r_on['n_orphans']})\n")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
