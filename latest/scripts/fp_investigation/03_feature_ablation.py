"""[fp_investigation] Trains THROWAWAY XGBoost models (nothing is saved under models/) with feature families removed.
    needs outputs/fp_live_flows.json from script 02.

Read-only experiments. Trains THROWAWAY models in memory; writes nothing under models/."""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
S = os.path.join(ROOT, "outputs")
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from train_xgboost_ae import FEATURE_COLS
from modules.trusted_model_loader import safe_load_joblib

df = pd.read_csv("NF-UNSW-NB15-v3.csv", usecols=FEATURE_COLS + ["Attack", "FLOW_START_MILLISECONDS"])
df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
y = np.where(df["Attack"].astype(str).str.strip().str.lower() == "benign", 0, 1)
X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
n = int(len(X) * 0.75)

live = json.load(open(os.path.join(S, "fp_live_flows.json")))
LV = np.array([r["vec"] for r in live], dtype=np.float32)
live_kind = []
for r in live:
    p = int(r["PROTOCOL"])
    if p == 2: live_kind.append("IGMP")
    elif p == 17 and int(r["L4_DST_PORT"]) == 53: live_kind.append("UDP-DNS")
    elif p == 17: live_kind.append("UDP-other(QUIC)")
    else: live_kind.append("TCP")
live_kind = np.array(live_kind)

def run(name, drop):
    cols = [c for c in FEATURE_COLS if c not in drop]
    idx = [FEATURE_COLS.index(c) for c in cols]
    m = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                          random_state=42, n_jobs=-1, eval_metric="logloss")
    m.fit(X.iloc[:n][cols], y[:n])
    pt = m.predict_proba(X.iloc[n:][cols])[:, 1]
    pred = (pt >= 0.5).astype(int)
    pl = m.predict_proba(pd.DataFrame(LV[:, idx], columns=cols))[:, 1]
    flagged = {k: f"{int((pl[live_kind == k] >= 0.75).sum())}/{int((live_kind == k).sum())}" for k in ("TCP", "UDP-DNS", "UDP-other(QUIC)", "IGMP")}
    imp = sorted(zip(cols, m.feature_importances_), key=lambda t: -t[1])[:3]
    print(f"\n[{name}] features={len(cols)}")
    print(f"  chrono-test  AUC={roc_auc_score(y[n:], pt):.4f}  F1={f1_score(y[n:], pred):.4f}  recall={recall_score(y[n:], pred):.4f}  precision={precision_score(y[n:], pred):.4f}")
    print(f"  live flows flagged (xgb>=0.75): {flagged}  | mean live prob={pl.mean():.3f}")
    print(f"  top importances: {[(a, round(float(b), 3)) for a, b in imp]}")

run("A. all 47 features (reproduces production)", [])
run("B. without MIN_IP_PKT_LEN, SHORTEST_FLOW_PKT", ["MIN_IP_PKT_LEN", "SHORTEST_FLOW_PKT"])
run("C. B + packet-size family (MAX/LONGEST, size buckets)", ["MIN_IP_PKT_LEN", "SHORTEST_FLOW_PKT", "MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"])
run("D. C + identity-like (ports, DNS id/ttl, protocol ids)", ["MIN_IP_PKT_LEN", "SHORTEST_FLOW_PKT", "MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "L4_SRC_PORT", "L4_DST_PORT", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "L7_PROTO", "ICMP_TYPE", "ICMP_IPV4_TYPE", "FTP_COMMAND_RET_CODE"])

# production model on live data, for reference
prod = xgb.XGBClassifier(); prod.load_model("models/xgboost_model.json")
pl = prod.predict_proba(LV)[:, 1]
print("\n[production model] live flagged:", {k: f"{int((pl[live_kind == k] >= 0.75).sum())}/{int((live_kind == k).sum())}" for k in ("TCP", "UDP-DNS", "UDP-other(QUIC)", "IGMP")})

# fusion layer weights
fu = safe_load_joblib("models/fusion_model.pkl")
print("fusion LR coef [xgb_prob, ae_norm]:", np.round(fu.coef_[0], 3), "intercept", np.round(fu.intercept_, 3))
