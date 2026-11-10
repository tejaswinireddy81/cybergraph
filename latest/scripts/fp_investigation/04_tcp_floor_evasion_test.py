"""[fp_investigation] Shows what the 52-byte TCP floor in flow_extractor.py does to a SYN scan.

Does the TCP 52-byte floor hide SYN-scan style flows? Read-only."""
import os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, xgboost as xgb
from scapy.all import Ether, IP, TCP
from flow_extractor import FlowExtractor
from train_xgboost_ae import FEATURE_COLS

m = xgb.XGBClassifier(); m.load_model("models/xgboost_model.json")
ex = FlowExtractor()
t = 2000.0
# A classic SYN scan: one bare 40-byte SYN to each of 30 ports, no reply (what hping/masscan style tools emit).
for i, port in enumerate(range(20, 50)):
    p = Ether()/IP(src="192.168.1.77", dst="192.168.1.37", ttl=64)/TCP(sport=40000 + i, dport=port, flags="S", seq=1)
    p.time = t + i * 0.001
    ex.add_packet(p)
recs = [ex._export(f) for f in list(ex._flows.values())]
recs = [r for r in recs if r]
print("exported SYN-scan flows:", len(recs))
raw_vals, floor_vals = [], []
mi = FEATURE_COLS.index("MIN_IP_PKT_LEN"); si = FEATURE_COLS.index("SHORTEST_FLOW_PKT")
for r in recs:
    row = np.array([[float(r.get(c, 0.0)) for c in FEATURE_COLS]], dtype=np.float32)
    floor_vals.append(float(m.predict_proba(row)[0, 1]))
    raw = row.copy(); raw[0, mi] = 40.0; raw[0, si] = 40.0      # what scapy actually saw on the wire
    raw_vals.append(float(m.predict_proba(raw)[0, 1]))
print("exported MIN_IP_PKT_LEN (after floor):", sorted({r['MIN_IP_PKT_LEN'] for r in recs}))
print("mean xgb prob WITH floor (current behaviour):    %.3f   flagged>=0.75: %d/%d" % (np.mean(floor_vals), sum(v >= .75 for v in floor_vals), len(floor_vals)))
print("mean xgb prob WITHOUT floor (true wire value 40): %.3f   flagged>=0.75: %d/%d" % (np.mean(raw_vals), sum(v >= .75 for v in raw_vals), len(raw_vals)))
