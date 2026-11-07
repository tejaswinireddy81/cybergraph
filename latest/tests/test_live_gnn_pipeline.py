"""
test_live_gnn_pipeline.py
==========================
End-to-End Test for the Live Flow Pipeline:
Live flow -> Update temporal graph -> Dynamic Temporal GNN -> GNN prediction -> Fusion / risk engine -> Dashboard

PHASE 1 FIX: sample_flow updated to include all 47 canonical features required by the
strict FeatureSchemaValidator (LiveTrafficEngine raises ValueError on missing features).
Original test had only ~20 keys, missing 36 of the 47 required NF-UNSW-NB15 features.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import torch
import numpy as np

from modules.live_traffic_engine import LiveTrafficEngine
from modules.online_graph_stream import OnlineGraphStream
from modules.realtime_alert_engine import RealtimeAlertEngine
from modules.dynamic_temporal_gnn import DynamicBipartiteTemporalGNN

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")
GNN_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_temporal_gnn.pt")
SCALERS_PATH = os.path.join(MODELS_DIR, "gnn_scalers.pkl")


def make_complete_sample_flow():
    """
    Returns a flow record with ALL 47 canonical NF-UNSW-NB15 features plus
    the required metadata (_src_ip, _dst_ip, etc.) for LiveTrafficEngine.

    PHASE 1 FIX: Original test had only ~20 keys — this was causing ValueError
    from the strict FeatureSchemaValidator introduced in the canonical pipeline.
    """
    return {
        # ── Metadata (not model features, but required by LiveTrafficEngine) ──
        "_src_ip": "192.168.1.105",
        "_dst_ip": "10.40.182.1",
        "_src_port": 54321,
        "_dst_port": 80,
        "_protocol": 6,
        "_pkt_count": 10.0,
        "_byte_count": 1500.0,
        "_duration": 50.0,
        "_pkt_rate": 200.0,
        "_timestamp": 1700000000.0,
        "_capture_latency_ms": 1.25,
        "IPV4_SRC_ADDR": "192.168.1.105",
        "IPV4_DST_ADDR": "10.40.182.1",
        "label_enc": 0,

        # ── 47 Canonical NF-UNSW-NB15 Features (FIX D applied: no MIN_TTL/MAX_TTL) ──
        "L4_SRC_PORT": 54321.0,
        "L4_DST_PORT": 80.0,
        "PROTOCOL": 6.0,
        "L7_PROTO": 7.0,           # HTTP
        "IN_BYTES": 1500.0,
        "IN_PKTS": 10.0,
        "OUT_BYTES": 800.0,
        "OUT_PKTS": 5.0,
        "TCP_FLAGS": 24.0,
        "CLIENT_TCP_FLAGS": 24.0,
        "SERVER_TCP_FLAGS": 18.0,
        "FLOW_DURATION_MILLISECONDS": 50.0,
        "DURATION_IN": 25.0,
        "DURATION_OUT": 25.0,
        "LONGEST_FLOW_PKT": 1460.0,
        "SHORTEST_FLOW_PKT": 40.0,
        "MIN_IP_PKT_LEN": 40.0,
        "MAX_IP_PKT_LEN": 1500.0,
        "SRC_TO_DST_SECOND_BYTES": 30000.0,
        "DST_TO_SRC_SECOND_BYTES": 16000.0,
        "RETRANSMITTED_IN_BYTES": 0.0,
        "RETRANSMITTED_IN_PKTS": 0.0,
        "RETRANSMITTED_OUT_BYTES": 0.0,
        "RETRANSMITTED_OUT_PKTS": 0.0,
        "SRC_TO_DST_AVG_THROUGHPUT": 240000.0,
        "DST_TO_SRC_AVG_THROUGHPUT": 128000.0,
        "NUM_PKTS_UP_TO_128_BYTES": 3.0,
        "NUM_PKTS_128_TO_256_BYTES": 1.0,
        "NUM_PKTS_256_TO_512_BYTES": 1.0,
        "NUM_PKTS_512_TO_1024_BYTES": 2.0,
        "NUM_PKTS_1024_TO_1514_BYTES": 3.0,
        "TCP_WIN_MAX_IN": 65535.0,
        "TCP_WIN_MAX_OUT": 65535.0,
        "ICMP_TYPE": 0.0,
        "ICMP_IPV4_TYPE": 0.0,
        "DNS_QUERY_ID": 0.0,
        "DNS_QUERY_TYPE": 0.0,
        "DNS_TTL_ANSWER": 0.0,
        "FTP_COMMAND_RET_CODE": 0.0,
        "SRC_TO_DST_IAT_MIN": 3.0,
        "SRC_TO_DST_IAT_MAX": 18.0,
        "SRC_TO_DST_IAT_AVG": 10.0,
        "SRC_TO_DST_IAT_STDDEV": 2.5,
        "DST_TO_SRC_IAT_MIN": 4.0,
        "DST_TO_SRC_IAT_MAX": 20.0,
        "DST_TO_SRC_IAT_AVG": 11.0,
        "DST_TO_SRC_IAT_STDDEV": 3.0,
    }


def test_end_to_end_live_pipeline():
    # 1. Initialize Engines
    traffic_engine = LiveTrafficEngine()
    alert_engine = RealtimeAlertEngine(enable_real_blocking=False)

    scalers = joblib.load(SCALERS_PATH)
    gnn_model = DynamicBipartiteTemporalGNN(host_dim=8, service_dim=8, edge_dim=9, hidden_dim=32)
    gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH))
    gnn_model.eval()

    stream = OnlineGraphStream(gnn_model, scalers, window_size=50)

    # 2. Simulate incoming live flow — must have all 47 canonical features
    sample_flow = make_complete_sample_flow()

    # Step 1: Live flow extraction & model scoring (XGBoost + AE + 2-signal Fusion)
    flow_rec = traffic_engine.process_flow_record(sample_flow)
    assert "fusion_score" in flow_rec, "flow_rec must contain fusion_score"
    assert "latency_breakdown" in flow_rec, "flow_rec must contain latency_breakdown"
    assert "xgb_prob" in flow_rec, "flow_rec must contain xgb_prob"
    assert "ae_score" in flow_rec, "flow_rec must contain ae_score"

    xgb_prob = flow_rec["xgb_prob"]
    ae_score = flow_rec["ae_score"]
    fusion_score = flow_rec["fusion_score"]
    assert 0.0 <= xgb_prob <= 1.0, f"xgb_prob out of range: {xgb_prob}"
    assert 0.0 <= ae_score <= 1.0, f"ae_score out of range: {ae_score}"
    assert 0.0 <= fusion_score <= 1.0, f"fusion_score out of range: {fusion_score}"

    latency = flow_rec["latency_breakdown"]["total_detection_ms"]
    assert latency < 15000, f"Latency too high: {latency} ms (expected < 15000 ms)"

    # Step 2: Update temporal graph
    stream.ingest_flow(flow_rec)

    # Steps 3 & 4: Dynamic Temporal GNN inference & edge risk prediction
    probs, pyg_data, h_map, s_map, edge_prob_map = stream.evaluate_realtime_gnn_risk()
    assert len(probs) > 0, "GNN must produce at least 1 edge probability"

    # Use edge probability map (deterministic, not probs[-1])
    sample_edge_key = ("HOST:192.168.1.105", "SERVICE:80/6")
    if sample_edge_key in edge_prob_map:
        gnn_pred = edge_prob_map[sample_edge_key]
    else:
        # Fallback: use first edge probability if exact key not present
        gnn_pred = float(probs[0])

    assert 0.0 <= gnn_pred <= 1.0, f"GNN edge probability out of range: {gnn_pred}"

    # Step 5: Fusion / risk engine evaluation
    eval_res = alert_engine.process_flow_eval(flow_rec, risk_score=gnn_pred, xgb_model=traffic_engine.xgb_model)
    assert "level" in eval_res, "eval_res must contain level"
    assert "risk_score" in eval_res, "eval_res must contain risk_score"

    print(f"\nEnd-to-End Pipeline Success!")
    print(f"  Live Flow Src IP      : {flow_rec['src_ip']}")
    print(f"  XGBoost Probability   : {xgb_prob:.6f}")
    print(f"  AE Norm MSE           : {ae_score:.6f}")
    print(f"  2-Signal Fusion Score : {fusion_score:.6f}")
    print(f"  GNN Edge Prediction   : {gnn_pred:.6f}")
    print(f"  Fusion Alert Level    : {eval_res['level']}")
    print(f"  Total Latency         : {latency:.2f} ms")


if __name__ == "__main__":
    test_end_to_end_live_pipeline()
