"""[fp_investigation] Scores a hand-built DNS exchange through the real FlowExtractor + XGBoost and attributes the score.

Read-only probe: score a realistic live DNS response and attribute the score."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, xgboost as xgb
from scapy.all import Ether, IP, UDP, DNS, DNSQR, DNSRR
from flow_extractor import FlowExtractor
from train_xgboost_ae import FEATURE_COLS
from modules.trusted_model_loader import safe_load_joblib

m = xgb.XGBClassifier(); m.load_model("models/xgboost_model.json")
scaler = safe_load_joblib("models/scaler.pkl")

def build_dns_flow(resp_pad=0):
    """One query 192.168.1.37 -> 8.8.4.4:53 and one response back, like a real resolver exchange."""
    ex = FlowExtractor()
    q = Ether()/IP(src="192.168.1.37", dst="8.8.4.4", ttl=128)/UDP(sport=55182, dport=53)/DNS(id=0x1a2b, rd=1, qd=DNSQR(qname="example.com"))
    r = Ether()/IP(src="8.8.4.4", dst="192.168.1.37", ttl=117)/UDP(sport=53, dport=55182)/DNS(
        id=0x1a2b, qr=1, rd=1, ra=1, qd=DNSQR(qname="example.com"),
        an=DNSRR(rrname="example.com", type="A", ttl=300, rdata="93.184.216.34"))
    t = 1000.0
    for p, dt in ((q, 0.0), (r, 0.021)):
        p.time = t + dt
        ex.add_packet(p)
    flows = ex.flush_expired() or []
    if not flows:
        # force-export remaining flows
        flows = [ex._export(f) for f in list(ex._flows.values())] if hasattr(ex, "_flows") else []
    return flows

flows = build_dns_flow()
print("flows exported:", len(flows))
for f in flows:
    row = np.array([[float(f.get(c, 0.0)) for c in FEATURE_COLS]], dtype=np.float32)
    p = float(m.predict_proba(row)[0, 1])
    print(f"\nflow {f.get('_src_ip')}:{f.get('L4_SRC_PORT')} -> {f.get('_dst_ip')}:{f.get('L4_DST_PORT')} proto={f.get('PROTOCOL')}  xgb_prob={p:.4f}")
    print("  MIN_IP_PKT_LEN=%s SHORTEST=%s MAX_IP=%s IN_PKTS=%s OUT_PKTS=%s IN_BYTES=%s OUT_BYTES=%s" % tuple(f.get(k) for k in ("MIN_IP_PKT_LEN","SHORTEST_FLOW_PKT","MAX_IP_PKT_LEN","IN_PKTS","OUT_PKTS","IN_BYTES","OUT_BYTES")))
    contrib = m.get_booster().predict(xgb.DMatrix(row, feature_names=FEATURE_COLS), pred_contribs=True)[0]
    order = np.argsort(-np.abs(contrib[:-1]))[:6]
    print("  top log-odds contributions:", [(FEATURE_COLS[i], round(float(contrib[i]), 2)) for i in order], "bias", round(float(contrib[-1]), 2))
    # counterfactual: only MIN_IP_PKT_LEN replaced
    for v in (52, 73):
        r2 = row.copy(); r2[0, FEATURE_COLS.index("MIN_IP_PKT_LEN")] = v
        print(f"  counterfactual MIN_IP_PKT_LEN={v}: xgb_prob={float(m.predict_proba(r2)[0,1]):.4f}")

# Model response curve vs MIN_IP_PKT_LEN (other features from the first flow)
if flows:
    base = np.array([[float(flows[0].get(c, 0.0)) for c in FEATURE_COLS]], dtype=np.float32)
    print("\nXGB prob as MIN_IP_PKT_LEN varies (all else = first DNS flow):")
    for v in (0, 28, 40, 44, 52, 60, 73, 76, 82, 90, 100, 128, 200):
        b = base.copy(); b[0, FEATURE_COLS.index("MIN_IP_PKT_LEN")] = v
        print(f"  {v:4d} -> {float(m.predict_proba(b)[0,1]):.4f}")
