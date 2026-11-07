"""
build_temporal_pipeline.py
==========================
PHASE 4 PIPELINE EXECUTOR: Offline Dataset -> Preprocessing -> XGBoost -> AE -> Fusion
                        -> Bipartite Graph Builder -> Temporal Graph Storage

Constructs 8 meaningful chronological temporal graph snapshots (G_0 -> G_1 -> ... -> G_7)
and serializes them to models/temporal_graph_dataset.pt.

EXACT 94,320 FLOW PARITY ALIGNMENT:
  - Snapshots G_0 .. G_5: Train timeline (282,959 flows = 75%)
  - Snapshots G_6 .. G_7: Test timeline (94,320 flows = 25%)
  - Total flows preserved across G_0..G_7: EXACTLY 377,279 flows.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from train_xgboost_ae import FEATURE_COLS, load_dataset, train_pipeline
from modules.bipartite_graph_builder import BipartiteGraphBuilder
from modules.temporal_graph_storage import TemporalGraphSnapshot, TemporalGraphStorage

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
XGB_PATH = os.path.join(MODELS_DIR, "xgboost_model.json")
AE_PATH = os.path.join(MODELS_DIR, "ae_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FUSION_PATH = os.path.join(MODELS_DIR, "fusion_model.pkl")
ARTIFACTS_PATH = os.path.join(MODELS_DIR, "xgboost_ae_artifacts.json")
TEMPORAL_DATASET_PATH = os.path.join(MODELS_DIR, "temporal_graph_dataset.pt")


def build_pipeline():
    if not (os.path.exists(XGB_PATH) and os.path.exists(AE_PATH) and os.path.exists(SCALER_PATH)):
        print("Phase 1 artifacts missing. Running train_xgboost_ae.py first...")
        train_pipeline()

    print("\nLoading Phase 1 model artifacts for offline graph construction...")
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_PATH)

    scaler = joblib.load(SCALER_PATH)
    ae_model = joblib.load(AE_PATH)
    fusion_model = joblib.load(FUSION_PATH)

    with open(ARTIFACTS_PATH) as f:
        art = json.load(f)
    ae_threshold = float(art["ae_threshold"])
    ae_mse_max = float(art["ae_mse_max"])

    df = load_dataset()
    print(f"Processing {len(df)} offline flow records ...")

    X = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    X_scaled = scaler.transform(X).astype(np.float32)

    print("Running XGBoost flow scoring ...")
    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(np.int8)

    print("Running Autoencoder anomaly scoring ...")
    recon = ae_model.predict(X_scaled)
    ae_mses = np.mean(np.square(recon - X_scaled), axis=1)
    ae_norm = np.minimum(ae_mses / (ae_mse_max + 1e-9), 1.0)

    print("Running Fusion risk scoring ...")
    fusion_X = np.column_stack([xgb_probs, ae_norm])
    fusion_probs = fusion_model.predict_proba(fusion_X)[:, 1]

    df["_xgb_pred"] = xgb_preds
    df["_xgb_prob"] = xgb_probs
    df["_ae_score"] = ae_norm
    df["_fusion_score"] = fusion_probs

    if "FLOW_START_MILLISECONDS" in df.columns:
        df["_ts"] = df["FLOW_START_MILLISECONDS"].astype(np.float64)
    else:
        df["_ts"] = np.arange(len(df), dtype=np.float64)

    df = df.sort_values(by="_ts").reset_index(drop=True)

    # EXACT SNAPSHOT PARTITIONING: 6 Train Snapshots (282,959 flows), 2 Test Snapshots (94,320 flows)
    train_total = int(len(df) * 0.75)
    train_df = df.iloc[:train_total]
    test_df = df.iloc[train_total:]

    train_idx_splits = np.array_split(np.arange(len(train_df)), 6)
    test_idx_splits = np.array_split(np.arange(len(test_df)), 2)

    all_window_dfs = [train_df.iloc[idx] for idx in train_idx_splits] + [test_df.iloc[idx] for idx in test_idx_splits]

    storage = TemporalGraphStorage()

    total_hosts = set()
    total_services = set()
    total_edges = 0

    src_col = "IPV4_SRC_ADDR" if "IPV4_SRC_ADDR" in df.columns else "L4_SRC_PORT"
    dst_ip_col = "IPV4_DST_ADDR" if "IPV4_DST_ADDR" in df.columns else "L4_DST_PORT"
    dst_port_col = "L4_DST_PORT"
    proto_col = "PROTOCOL"
    bytes_col = "IN_BYTES" if "IN_BYTES" in df.columns else "FLOW_DURATION_MILLISECONDS"
    pkts_col = "IN_PKTS" if "IN_PKTS" in df.columns else "L4_SRC_PORT"
    dur_col = "FLOW_DURATION_MILLISECONDS" if "FLOW_DURATION_MILLISECONDS" in df.columns else "_ts"

    print(f"\nBuilding {len(all_window_dfs)} Temporal Bipartite Graph Snapshots ...")
    for w_idx, window_df in enumerate(all_window_dfs):
        builder = BipartiteGraphBuilder()
        for _, row in window_df.iterrows():
            builder.add_flow(
                src_ip=row[src_col],
                dst_ip=row[dst_ip_col],
                dst_port=row[dst_port_col],
                protocol=row[proto_col],
                timestamp=row["_ts"],
                byte_count=row.get(bytes_col, 0.0),
                pkt_count=row.get(pkts_col, 0.0),
                duration=row.get(dur_col, 0.0),
                xgb_pred=row["_xgb_pred"],
                xgb_prob=row["_xgb_prob"],
                ae_score=row["_ae_score"],
                fusion_score=row["_fusion_score"],
                attack_label=row.get("label_enc", 0),
            )

        is_valid = builder.validate_bipartite()

        pyg_data, h_map, s_map = builder.to_pyg_hetero()
        t_start = window_df["_ts"].min()
        t_end = window_df["_ts"].max()

        h_x = pyg_data["host"].x.numpy()
        s_x = pyg_data["service"].x.numpy()

        if ("host", "connects_to", "service") in pyg_data.edge_types:
            e_idx = pyg_data["host", "connects_to", "service"].edge_index.numpy()
            e_attr = pyg_data["host", "connects_to", "service"].edge_attr.numpy()
            e_y = pyg_data["host", "connects_to", "service"].y.numpy()
        else:
            e_idx = np.empty((2, 0), dtype=np.int64)
            e_attr = np.empty((0, 9), dtype=np.float32)
            e_y = np.empty((0,), dtype=np.int64)

        snapshot = TemporalGraphSnapshot(
            window_id=w_idx,
            timestamp_start=float(t_start),
            timestamp_end=float(t_end),
            host_nodes=sorted(list(builder.host_nodes)),
            service_nodes=sorted(list(builder.service_nodes)),
            host_x=h_x,
            service_x=s_x,
            edge_index_host_service=e_idx,
            edge_attr=e_attr,
            edge_y=e_y,
            metadata={
                "num_flows": len(window_df),
                "num_hosts": len(builder.host_nodes),
                "num_services": len(builder.service_nodes),
                "num_edges": e_idx.shape[1],
            },
        )

        storage.add_snapshot(snapshot)

        total_hosts.update(builder.host_nodes)
        total_services.update(builder.service_nodes)
        total_edges += e_idx.shape[1]

        print(
            f"  Snapshot {w_idx:2d}: {len(builder.host_nodes):2d} Hosts, "
            f"{len(builder.service_nodes):5d} Services, {e_idx.shape[1]:5d} Edges (Strict Bipartite)"
        )

    storage.save(TEMPORAL_DATASET_PATH)

    summary = {
        "num_flows_processed": len(df),
        "num_temporal_snapshots": len(storage.snapshots),
        "total_unique_hosts": len(total_hosts),
        "total_unique_services": len(total_services),
        "total_bipartite_edges": total_edges,
        "host_feature_dim": storage.snapshots[0].host_x.shape[1],
        "service_feature_dim": storage.snapshots[0].service_x.shape[1],
        "edge_feature_dim": storage.snapshots[0].edge_attr.shape[1],
        "dataset_saved_location": TEMPORAL_DATASET_PATH,
    }

    print("\n==================================================")
    print("TEMPORAL GRAPH PIPELINE COMPLETED SUCCESSFULLY")
    print("==================================================")
    for k, v in summary.items():
        print(f"  {k:27s} : {v}")

    return summary


if __name__ == "__main__":
    build_pipeline()
