"""
train_temporal_gnn.py
=====================
PHASE 5 TRAINING ENGINE: Dynamic Bipartite Temporal GNN (Scaled & Calibrated)

Trains the DynamicBipartiteTemporalGNN model on the temporal graph dataset sequence (G_0 -> G_1 -> ... -> G_7).

Applies Feature Scaling on node/edge attributes & Strict Temporal Snapshot Splitting:
  - Train Snapshots: G_0 through G_5 (First 75% of flow timeline, 282,959 flows)
  - Test Snapshots : G_6 through G_7 (Final 25% of flow timeline, 94,320 flows)

Saves trained model weights to:
  - models/dynamic_temporal_gnn.pt
  - models/gnn_scalers.pkl
  - outputs/gnn_training_report.json
"""

import os
import json
import random
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from modules.temporal_graph_storage import TemporalGraphStorage
from modules.dynamic_temporal_gnn import DynamicBipartiteTemporalGNN

# --- Reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TEMPORAL_DATASET_PATH = os.path.join(MODELS_DIR, "temporal_graph_dataset.pt")
GNN_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_temporal_gnn.pt")
SCALERS_PATH = os.path.join(MODELS_DIR, "gnn_scalers.pkl")
GNN_REPORT_PATH = os.path.join(OUT_DIR, "gnn_training_report.json")


def log(msg=""):
    print(msg, flush=True)


def train_gnn():
    log("==================================================")
    log("PHASE 5: TRAINING DYNAMIC BIPARTITE TEMPORAL GNN")
    log("==================================================")

    if not os.path.exists(TEMPORAL_DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset missing at '{TEMPORAL_DATASET_PATH}'. Please run 'python build_temporal_pipeline.py' first."
        )

    storage = TemporalGraphStorage.load(TEMPORAL_DATASET_PATH)
    num_snapshots = len(storage.snapshots)
    log(f"Loaded {num_snapshots} temporal graph snapshots.")

    # Chronological Temporal Snapshot Split: 75% Train (0..5), 25% Test (6..7)
    train_split_idx = int(round(num_snapshots * 0.75))
    train_snapshots = storage.snapshots[:train_split_idx]
    test_snapshots = storage.snapshots[train_split_idx:]

    log(f"Train Snapshots: G_0 .. G_{train_split_idx-1} ({len(train_snapshots)} snapshots)")
    log(f"Test Snapshots : G_{train_split_idx} .. G_{num_snapshots-1} ({len(test_snapshots)} snapshots)")

    # 1. Fit StandardScalers on Host features, Service features, and Edge attributes from Train Snapshots
    all_train_host_x = np.vstack([s.host_x for s in train_snapshots if s.host_x.shape[0] > 0])
    all_train_service_x = np.vstack([s.service_x for s in train_snapshots if s.service_x.shape[0] > 0])
    all_train_edge_attr = np.vstack([s.edge_attr for s in train_snapshots if s.edge_attr.shape[0] > 0])

    host_scaler = StandardScaler().fit(all_train_host_x)
    service_scaler = StandardScaler().fit(all_train_service_x)
    edge_scaler = StandardScaler().fit(all_train_edge_attr)

    joblib.dump({
        "host_scaler": host_scaler,
        "service_scaler": service_scaler,
        "edge_scaler": edge_scaler,
    }, SCALERS_PATH)

    log("Fit and saved StandardScalers for node features and edge attributes.")

    # Model initialization
    sample_snap = storage.snapshots[0]
    host_dim = sample_snap.host_x.shape[1]
    service_dim = sample_snap.service_x.shape[1]
    edge_dim = sample_snap.edge_attr.shape[1]

    model = DynamicBipartiteTemporalGNN(
        host_dim=host_dim,
        service_dim=service_dim,
        edge_dim=edge_dim,
        hidden_dim=32,
    )
    optimizer = Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    # Count class distribution across training snapshots for class weighting
    all_train_y = np.concatenate([s.edge_y for s in train_snapshots if len(s.edge_y) > 0])
    n_pos = int((all_train_y == 1).sum())
    n_neg = int((all_train_y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    log(f"\nTraining Class Balance -> Benign: {n_neg} | Attack: {n_pos} | PosWeight: {pos_weight.item():.2f}")

    # Training Loop over Temporal Sequence Epochs
    epochs = 50
    log(f"\nTraining Dynamic Temporal GNN over {epochs} sequence epochs ...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        total_edges_trained = 0

        # Temporal sequence propagation over time steps G_0 -> G_1 -> ... -> G_5
        prev_h, prev_s = None, None
        optimizer.zero_grad()

        for s_idx, snap in enumerate(train_snapshots):
            if snap.edge_index_host_service.shape[1] == 0:
                continue

            hx_scaled = torch.tensor(host_scaler.transform(snap.host_x), dtype=torch.float32)
            sx_scaled = torch.tensor(service_scaler.transform(snap.service_x), dtype=torch.float32)
            eidx = torch.tensor(snap.edge_index_host_service, dtype=torch.long)
            eattr_scaled = torch.tensor(edge_scaler.transform(snap.edge_attr), dtype=torch.float32)
            targets = torch.tensor(snap.edge_y, dtype=torch.float32)

            logits, prev_h, prev_s = model(hx_scaled, sx_scaled, eidx, eattr_scaled, prev_h, prev_s)

            loss = criterion(logits, targets)
            loss.backward()

            prev_h = prev_h.detach()
            prev_s = prev_s.detach()

            epoch_loss += loss.item() * len(targets)
            total_edges_trained += len(targets)

        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / max(total_edges_trained, 1)
            log(f"  Epoch {epoch+1:2d}/{epochs:2d} - Sequence Loss: {avg_loss:.6f}")

    # Save trained weights
    torch.save(model.state_dict(), GNN_MODEL_PATH)
    log(f"\nTrained model weights saved to: {GNN_MODEL_PATH}")

    # ============================================================
    # EVALUATION ON UNSEEN FUTURE TEMPORAL TEST SNAPSHOTS (G_6, G_7)
    # ============================================================
    log("\nEvaluating Dynamic Temporal GNN on future test snapshots (G_6 .. G_7) ...")
    model.eval()

    test_preds_list = []
    test_probs_list = []
    test_targets_list = []

    # Warm up temporal states with train sequence
    prev_h, prev_s = None, None
    with torch.no_grad():
        for snap in train_snapshots:
            if snap.edge_index_host_service.shape[1] == 0:
                continue
            hx_scaled = torch.tensor(host_scaler.transform(snap.host_x), dtype=torch.float32)
            sx_scaled = torch.tensor(service_scaler.transform(snap.service_x), dtype=torch.float32)
            eidx = torch.tensor(snap.edge_index_host_service, dtype=torch.long)
            eattr_scaled = torch.tensor(edge_scaler.transform(snap.edge_attr), dtype=torch.float32)
            _, prev_h, prev_s = model(hx_scaled, sx_scaled, eidx, eattr_scaled, prev_h, prev_s)

        # Predict on future test sequence G_6 -> G_7
        for snap in test_snapshots:
            if snap.edge_index_host_service.shape[1] == 0:
                continue

            hx_scaled = torch.tensor(host_scaler.transform(snap.host_x), dtype=torch.float32)
            sx_scaled = torch.tensor(service_scaler.transform(snap.service_x), dtype=torch.float32)
            eidx = torch.tensor(snap.edge_index_host_service, dtype=torch.long)
            eattr_scaled = torch.tensor(edge_scaler.transform(snap.edge_attr), dtype=torch.float32)
            targets = snap.edge_y

            logits, prev_h, prev_s = model(hx_scaled, sx_scaled, eidx, eattr_scaled, prev_h, prev_s)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(np.int8)

            test_preds_list.append(preds)
            test_probs_list.append(probs)
            test_targets_list.append(targets)

    all_test_preds = np.concatenate(test_preds_list)
    all_test_probs = np.concatenate(test_probs_list)
    all_test_targets = np.concatenate(test_targets_list)

    acc = float(accuracy_score(all_test_targets, all_test_preds))
    prec = float(precision_score(all_test_targets, all_test_preds, zero_division=0))
    rec = float(recall_score(all_test_targets, all_test_preds, zero_division=0))
    f1 = float(f1_score(all_test_targets, all_test_preds, zero_division=0))
    auc = float(roc_auc_score(all_test_targets, all_test_probs))
    cm = confusion_matrix(all_test_targets, all_test_preds).tolist()

    log("\n==================================================")
    log("DYNAMIC TEMPORAL GNN — TEST EVALUATION RESULTS")
    log("==================================================")
    log(f"Accuracy : {acc:.6f}")
    log(f"Precision: {prec:.6f}")
    log(f"Recall   : {rec:.6f}")
    log(f"F1 Score : {f1:.6f}")
    log(f"ROC-AUC  : {auc:.6f}")
    log(f"Confusion Matrix: {cm}")

    report = {
        "model_architecture": "DynamicBipartiteTemporalGNN",
        "num_train_snapshots": len(train_snapshots),
        "num_test_snapshots": len(test_snapshots),
        "test_edges_evaluated": len(all_test_targets),
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm,
        },
    }

    with open(GNN_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    log(f"\nReport saved to: {GNN_REPORT_PATH}")
    return report


if __name__ == "__main__":
    train_gnn()
