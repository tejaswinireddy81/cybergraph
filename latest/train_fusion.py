"""
train_fusion.py
================
Replaces the original hand-picked fixed weights (rf: 0.30, ae: 0.20,
gnn: 0.25, gat: 0.25) with a LEARNED logistic-regression fusion layer,
trained on real held-out flows from NF-UNSW-NB15-v3. This is the
standard "stacking" pattern: combine the flow-level detectors (RF,
Autoencoder) with the host-level structural signal (GAT) into one
calibrated final probability, instead of asserting weights by hand.

Run after train_models.py AND train_gnn.py:
    python train_fusion.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from modules.gat import GATModel
from train_gnn import build_host_graph

warnings.filterwarnings("ignore")
np.random.seed(42)

HERE = os.path.dirname(__file__)
DATASET_PATH = os.path.join(HERE, "NF-UNSW-NB15-v3.csv")
MODELS_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "outputs")

# duplicated from train_models.py (NOT imported — that module runs training
# as top-level code with no __main__ guard, so importing it would re-run
# the entire RF+AE training pipeline as a side effect)
FEATURE_COLS = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    # FIX D: MIN_TTL and MAX_TTL REMOVED — severe dataset TTL bias
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]


def main():
    print("Loading dataset & trained artifacts …")
    df = pd.read_csv(DATASET_PATH)

    rf = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "artifacts.json")) as f:
        art = json.load(f)
    ae_threshold = art["ae_threshold"]
    ae_mse_max = art["ae_mse_max_train"]

    import tensorflow as tf
    ae = tf.keras.models.load_model(os.path.join(MODELS_DIR, "ae_model.keras"))

    with open(os.path.join(MODELS_DIR, "gat_artifacts.json")) as f:
        gat_art = json.load(f)
    gat = GATModel(input_dim=gat_art["input_dim"], hidden_dim=gat_art["hidden_dim"], output_dim=2)
    gat.load_state_dict(torch.load(os.path.join(MODELS_DIR, "gat_model.pt"), map_location="cpu"))
    gat.eval()
    graph_scaler = joblib.load(os.path.join(MODELS_DIR, "graph_feature_scaler.pkl"))

    # --- rebuild the same host graph used at GAT training time, and score every host ---
    print("Scoring host graph with trained GAT …")
    nodes, feats, labels, edge_index = build_host_graph(df)
    feats_scaled = graph_scaler.transform(feats).astype(np.float32)
    from torch_geometric.data import Data
    data = Data(x=torch.tensor(feats_scaled), edge_index=edge_index)
    with torch.no_grad():
        gat_probs_per_host = F.softmax(gat(data), dim=1)[:, 1].numpy()
    host_gat_prob = dict(zip(nodes, gat_probs_per_host))
    default_gat_prob = float(np.median(gat_probs_per_host))  # for any host missing from the graph

    # --- SAME train/test flow split as train_models.py, so we only fuse on the model's held-out data ---
    df["label_enc"] = np.where(
        df["Attack"].astype(str).str.strip().str.lower() == "benign", 0, 1).astype("int8")
    X = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = df["label_enc"]
    src_ips = df["IPV4_SRC_ADDR"]

    _, X_test, _, y_test, _, src_test = train_test_split(
        X, y, src_ips, test_size=0.25, random_state=42, stratify=y)

    print(f"Fusion training pool (held-out flows): {X_test.shape}")

    # --- compute the 3 base signals per flow ---
    rf_attack_idx = int(np.where(rf.classes_ == 1)[0][0])
    rf_probs = rf.predict_proba(X_test)[:, rf_attack_idx]

    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    recon = ae.predict(X_test_scaled, batch_size=4096, verbose=0)
    ae_mse = np.mean(np.square(recon - X_test_scaled), axis=1)
    ae_norm = np.minimum(ae_mse / (ae_mse_max + 1e-9), 1.0)

    gat_probs = np.array([host_gat_prob.get(ip, default_gat_prob) for ip in src_test])

    fusion_X = np.column_stack([rf_probs, ae_norm, gat_probs])
    fusion_y = y_test.values

    # --- further split so fusion model is evaluated on flows it never trained on ---
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        fusion_X, fusion_y, test_size=0.3, random_state=42, stratify=fusion_y)

    print("\nTraining logistic-regression fusion layer …")
    fusion = LogisticRegression(class_weight="balanced", max_iter=1000)
    fusion.fit(Xf_train, yf_train)

    yf_pred = fusion.predict(Xf_test)
    yf_prob = fusion.predict_proba(Xf_test)[:, 1]
    report = classification_report(yf_test, yf_pred, target_names=["Normal", "Attack"])
    auc = roc_auc_score(yf_test, yf_prob)

    print("\n=== Fusion layer — held-out flows ===")
    print(report)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"\nLearned coefficients (rf, ae, gat): {fusion.coef_[0]}")
    print(f"Intercept: {fusion.intercept_[0]:.4f}")

    joblib.dump(fusion, os.path.join(MODELS_DIR, "fusion_model.pkl"))
    with open(os.path.join(OUT_DIR, "fusion_report.txt"), "w") as f:
        f.write("=== Learned Fusion Layer (replaces fixed hand-picked weights) ===\n\n")
        f.write(f"Features: [rf_prob, ae_normalized_error, gat_host_prob]\n")
        f.write(f"Learned coefficients: {fusion.coef_[0].tolist()}\n")
        f.write(f"Intercept: {fusion.intercept_[0]:.4f}\n\n")
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.4f}\n")

    print(f"\nSaved fusion model -> {MODELS_DIR}/fusion_model.pkl")
    return auc


if __name__ == "__main__":
    main()
