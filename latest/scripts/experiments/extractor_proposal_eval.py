"""
scripts/experiments/extractor_proposal_eval.py
==============================================
Evidence for docs/extractor_investigation_and_proposal.md. Compares the PRODUCTION extractor with the CURRENT extractor
(flags, defaults = original behaviour) (nothing in production is modified; nothing is trained; models are only used for inference).

  E0  equivalence      prototype with default flags == production, bit for bit, on the same packet streams
  E1  orphan flows     flows per TCP connection, packets-per-flow, FIN/bidirectional shares vs the training data
  E2  timestamps       sensitivity of duration / IAT features to processing delay: wall clock vs capture timestamp
  E3  model effect     production XGBoost alerts on benign and per-family attacks, for the 4 combinations
                       {legacy, proposed} x {TCP floor on, off}
  E4  thread safety    capture thread vs flush thread hammering one extractor

All traffic is SYNTHETIC (evaluation/synthetic_traffic.py): a model of traffic, so every number here is provisional
until lab captures are approved. The processing-delay model in E2 is an assumption, not a measurement.

    python scripts/experiments/extractor_proposal_eval.py
"""
import importlib.util
import json
import os
import sys
import threading
import time
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

import flow_extractor as PROP  # noqa: E402   current extractor (has the tcp_close_mode / timestamp_source flags)
from evaluation.synthetic_traffic import DEFAULT_COUNTS, generate  # noqa: E402
from train_xgboost_ae import FEATURE_COLS  # noqa: E402

# frozen, verbatim copy of the extractor BEFORE the flags: the "as it was" reference
spec = importlib.util.spec_from_file_location("flow_extractor_original",
                                              os.path.join(ROOT, "tests", "golden", "flow_extractor_original.py"))
PROD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROD)

FACTORIES = {
    "original": (lambda floor: PROD.FlowExtractor(), [PROD]),       # frozen pre-flag extractor (floor fixed at 52)
    "production": (lambda floor: PROP.FlowExtractor(min_pkt_floor_tcp=floor), [PROP]),   # current extractor, DEFAULT flags
    "proposed": (lambda floor: PROP.FlowExtractor(min_pkt_floor_tcp=floor, tcp_close_mode="linger",
                                                  timestamp_source="capture"), [PROP]),
    "capture-only": (lambda floor: PROP.FlowExtractor(min_pkt_floor_tcp=floor, timestamp_source="capture"), [PROP]),
    "linger-only": (lambda floor: PROP.FlowExtractor(min_pkt_floor_tcp=floor, tcp_close_mode="linger"), [PROP]),
}


def gen(name, seed=11, floor=52, **kw):
    factory, mods = FACTORIES[name]
    return generate(seed=seed, n_sessions=3, floor=floor, extractor_factory=factory, clock_modules=mods, **kw)


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:5.1f}%"


# UNSW-NB15-v3 (local file), benign TCP flows, measured in this study
TRAIN = {"single_packet": 0.0000, "both_fin": 0.745, "bidirectional": 1.000, "rst": 0.0000,
         "pkts_quantiles": {"p5": 8, "p50": 36, "p95": 248}}


def tcp_stats(df):
    t = df[(df.PROTOCOL == 6) & (df._label == 0)]
    pk = t.IN_PKTS + t.OUT_PKTS
    flags_c, flags_s = t.CLIENT_TCP_FLAGS.astype(int), t.SERVER_TCP_FLAGS.astype(int)
    return {
        "n_flows": int(len(t)),
        "single_packet": float((pk <= 1).mean()), "le2_packets": float((pk <= 2).mean()),
        "bidirectional": float(((t.IN_PKTS > 0) & (t.OUT_PKTS > 0)).mean()),
        "both_fin": float((((flags_c & 1) > 0) & ((flags_s & 1) > 0)).mean()),
        "rst": float((((t.TCP_FLAGS.astype(int)) & 4) > 0).mean()),
        "pkts_p5_p50_p95": [float(v) for v in pk.quantile([.05, .5, .95])],
    }


def main():
    out = {}
    print("=" * 100 + "\nE0  EQUIVALENCE: current extractor (default flags) vs the frozen original (same packets, same seed)")
    a, b = gen("original"), gen("production")
    cols = FEATURE_COLS + ["_timestamp", "_pkt_count", "_byte_count", "_src_ip", "_dst_ip", "_src_port", "_dst_port"]
    same = a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))
    print(f"  flows: production={len(a)} current={len(b)} | all {len(cols)} compared columns identical: {same}")
    out["E0_equivalent"] = bool(same)

    print("\n" + "=" * 100 + "\nE1  ORPHAN FLOWS (TCP, benign synthetic traffic)")
    n_conn = 3 * sum(DEFAULT_COUNTS[c] for c in ("https_tcp", "http_tcp", "other_tcp", "dns_tcp"))
    rows = {}
    for name in ("production", "linger-only", "proposed"):
        df = a if name == "production" else gen(name)
        s = tcp_stats(df)
        s["flows_per_connection"] = s["n_flows"] / n_conn
        rows[name] = s
    print(f"  TCP connections generated: {n_conn}; training data (UNSW benign TCP): single-packet {pct(TRAIN['single_packet'])}, "
          f"both-FIN {pct(TRAIN['both_fin'])}, bidirectional {pct(TRAIN['bidirectional'])}, RST {pct(TRAIN['rst'])}")
    print(f"  {'extractor':14s}{'flows':>7s}{'flows/conn':>11s}{'1-pkt':>8s}{'<=2pkt':>8s}{'bidir':>8s}{'both-FIN':>10s}   pkts p5/p50/p95")
    for name, s in rows.items():
        print(f"  {name:14s}{s['n_flows']:7d}{s['flows_per_connection']:11.2f}{pct(s['single_packet']):>8s}{pct(s['le2_packets']):>8s}"
              f"{pct(s['bidirectional']):>8s}{pct(s['both_fin']):>10s}   {s['pkts_p5_p50_p95']}")
    out["E1"] = rows

    print("\n" + "=" * 100 + "\nE2  TIMESTAMPS: error of duration / IAT vs ground truth when processing lags capture (ASSUMED delay model)")
    truth = gen("production")                         # processing cost 0 => wall clock == capture clock
    key = ["_session", "_src_ip", "_src_port", "_dst_ip", "_dst_port", "PROTOCOL"]

    def aligned(df):
        d = df.sort_values("_timestamp").copy()
        d["_k"] = d.groupby(key).cumcount()
        return d.set_index(key + ["_k"])

    T = aligned(truth)
    e2 = {}
    for cost, jitter in ((0.0002, 0.0001), (0.0005, 0.0003), (0.002, 0.001)):
        for name in ("production", "capture-only"):
            d = aligned(gen(name, proc_cost_s=cost, proc_jitter_s=jitter))
            j = T.join(d, how="inner", lsuffix="_t", rsuffix="_d")
            for feat in ("FLOW_DURATION_MILLISECONDS", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_AVG_THROUGHPUT"):
                err = (j[feat + "_d"] - j[feat + "_t"]).abs() / np.maximum(j[feat + "_t"].abs(), 1.0)
                e2.setdefault(f"{cost * 1000:.1f}ms+{jitter * 1000:.1f}ms jitter", {}).setdefault(name, {})[feat] = {
                    "median_rel_err": float(err.median()), "share_over_10pct": float((err > 0.10).mean()), "n": int(len(j))}
    print(f"  (relative error vs the capture-time truth, over {len(T)} synthetic flows; delay model: per-packet cost + exponential jitter)")
    print(f"  {'processing delay':22s}{'extractor':14s}{'duration: med / >10%':>24s}{'IAT avg: med / >10%':>24s}{'throughput: med / >10%':>26s}")
    for setting, per in e2.items():
        for name, feats in per.items():
            f = lambda k: f"{pct(feats[k]['median_rel_err'])} / {pct(feats[k]['share_over_10pct'])}"  # noqa: E731
            print(f"  {setting:22s}{name:14s}{f('FLOW_DURATION_MILLISECONDS'):>24s}{f('SRC_TO_DST_IAT_AVG'):>24s}{f('SRC_TO_DST_AVG_THROUGHPUT'):>26s}")
    out["E2"] = e2

    print("\n" + "=" * 100 + "\nE3  MODEL EFFECT (production XGBoost F0, alert at 0.75; synthetic traffic, PROVISIONAL)")
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(ROOT, "models", "xgboost_model.json"))
    e3 = {}
    for name in ("production", "proposed"):
        for floor in (52, 0):
            df = gen(name, floor=floor)
            p = model.predict_proba(df[FEATURE_COLS].astype(np.float32))[:, 1]
            al = p >= 0.75
            y = df._label.to_numpy().astype(int)
            tcp = (df.PROTOCOL == 6).to_numpy()
            win = (df._session == "synthetic-s0").to_numpy()          # session 0 = Windows client (bare 40-byte ACKs)
            r = {"flows": int(len(df)), "benign_FPR": float(al[y == 0].mean()), "benign_TCP_FPR": float(al[(y == 0) & tcp].mean()),
                 "TCP_FPR_win": float(al[(y == 0) & tcp & win].mean()), "TCP_FPR_unix": float(al[(y == 0) & tcp & ~win].mean()),
                 "benign_nonTCP_FPR": float(al[(y == 0) & ~tcp].mean()), "recall_all": float(al[y == 1].mean())}
            for fam in sorted(set(df.loc[y == 1, "_family"])):
                r[f"recall_{fam}"] = float(al[(df._family == fam).to_numpy()].mean())
            e3[f"{name}/floor{'ON' if floor else 'OFF'}"] = r
    cols3 = list(next(iter(e3.values())))[1:]
    short = lambda c: c.replace("benign_", "b_").replace("recall_", "rec_").replace("nonTCP", "nTCP")  # noqa: E731
    print(f"  {'configuration':22s}" + "".join(f"{short(c):>12s}" for c in cols3))
    for k, r in e3.items():
        print(f"  {k:22s}" + "".join(f"{pct(r[c]):>12s}" for c in cols3))
    out["E3"] = e3

    print("\n" + "=" * 100 + "\nE4  THREAD SAFETY: capture thread feeding packets while a flush thread runs concurrently")
    from scapy.all import IP, TCP, Ether

    def stress(mod, seconds=3.0):
        ex = mod.FlowExtractor(idle_timeout_s=0.0, active_timeout_s=0.0)
        stop, errors, exported = threading.Event(), [], []

        def flusher():
            while not stop.is_set():
                try:
                    exported.extend(ex.flush_expired())
                except Exception as exc:               # live_capture swallows these silently
                    errors.append(type(exc).__name__)

        th = threading.Thread(target=flusher)
        th.start()
        n = 0
        t_end = time.time() + seconds
        while time.time() < t_end:
            sp = 1024 + (n % 60000)
            for flags in ("S", "A", "FA"):
                ex.add_packet(Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=sp, dport=80, flags=flags))
            n += 1
        stop.set()
        th.join()
        exported.extend(ex.flush_expired())
        errors_in_add = ex.pkt_errors
        return {"connections": n, "exported_flows": len(exported), "flusher_exceptions": len(errors),
                "add_packet_errors": errors_in_add}
    e4 = {}
    for label, mod in (("original (no lock)", PROD), ("current (lock)", PROP)):
        runs = [stress(mod) for _ in range(4)]
        e4[label] = {"runs": len(runs), "connections": sum(r["connections"] for r in runs),
                     "add_packet_errors": sum(r["add_packet_errors"] for r in runs),
                     "flusher_exceptions": sum(r["flusher_exceptions"] for r in runs)}
        print(f"  {label:18s} {e4[label]}")
    out["E4"] = e4

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "outputs", "extractor_proposal_eval.json"), "w"), indent=2, default=float)
    print("\nwrote outputs/extractor_proposal_eval.json")


if __name__ == "__main__":
    main()
