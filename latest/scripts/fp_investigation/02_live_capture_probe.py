"""[fp_investigation] Sniffs real packets (needs Npcap/admin, like the app), generates DNS/HTTP traffic, scores every flow.
    usage: python scripts/fp_investigation/02_live_capture_probe.py [seconds] [out.json]  (default out: outputs/fp_live_flows.json)

Read-only live probe. Sniffs real packets (like live_capture.py), builds flows with the
real FlowExtractor, scores each with the trained XGBoost and records full feature vectors."""
import os, sys, json, time, threading, warnings, subprocess
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, xgboost as xgb
from scapy.sendrecv import sniff
from flow_extractor import FlowExtractor
from train_xgboost_ae import FEATURE_COLS

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "outputs", "fp_live_flows.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
m = xgb.XGBClassifier(); m.load_model("models/xgboost_model.json")
ex = FlowExtractor()
lock = threading.Lock()
rows = []
stop = threading.Event()

def score(rec):
    row = np.array([[float(rec.get(c, 0.0)) for c in FEATURE_COLS]], dtype=np.float32)
    p = float(m.predict_proba(row)[0, 1])
    contrib = m.get_booster().predict(xgb.DMatrix(row, feature_names=FEATURE_COLS), pred_contribs=True)[0]
    top = np.argsort(-np.abs(contrib[:-1]))[:4]
    d = {k: rec.get(k) for k in ("_src_ip", "_dst_ip", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "IN_PKTS", "OUT_PKTS",
                                  "IN_BYTES", "OUT_BYTES", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SHORTEST_FLOW_PKT",
                                  "FLOW_DURATION_MILLISECONDS", "DNS_TTL_ANSWER", "DNS_QUERY_ID", "TCP_FLAGS")}
    d["xgb_prob"] = p
    d["top"] = [(FEATURE_COLS[i], round(float(contrib[i]), 2)) for i in top]
    d["vec"] = [float(x) for x in row[0]]
    with lock:
        rows.append(d)

def flusher():
    while not stop.is_set():
        time.sleep(2)
        with lock:
            exp = ex.flush_expired()
        for r in exp:
            score(r)

def on_pkt(pkt):
    with lock:
        r = ex.add_packet(pkt)
    if r:
        score(r)

def traffic():
    time.sleep(3)
    for dom in ("example.com", "wikipedia.org", "github.com", "python.org", "microsoft.com", "bbc.co.uk", "mozilla.org", "apple.com"):
        subprocess.run(["nslookup", dom, "8.8.4.4"], capture_output=True, timeout=15)
        subprocess.run(["nslookup", dom, "8.8.8.8"], capture_output=True, timeout=15)
        time.sleep(1)
    subprocess.run(["curl", "-s", "-o", "NUL", "--max-time", "8", "https://example.com"], capture_output=True)
    subprocess.run(["curl", "-s", "-o", "NUL", "--max-time", "8", "http://neverssl.com"], capture_output=True)

threading.Thread(target=flusher, daemon=True).start()
threading.Thread(target=traffic, daemon=True).start()
end = time.time() + DURATION
while time.time() < end:
    sniff(prn=on_pkt, store=False, timeout=1.5)
stop.set()
with lock:
    exp = ex.flush_expired()
    for key in list(ex._flows):
        r = ex._export(ex._flows[key])
        if r: exp.append(r)
for r in exp:
    score(r)
json.dump(rows, open(OUT, "w"))
print("flows scored:", len(rows))
