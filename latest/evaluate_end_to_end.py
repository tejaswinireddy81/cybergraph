"""
evaluate_end_to_end.py
======================

End-to-end evaluation of the CyberGraph detection pipeline.

Evaluates:
    RF
    Autoencoder
    GAT
    Learned Fusion

Uses the same held-out flow split used by train_fusion.py.

Outputs:
    - Accuracy
    - Precision
    - Recall
    - F1
    - ROC-AUC
    - Confusion Matrix
    - Per-model comparison
    - Fusion improvement/difference
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from modules.trusted_model_loader import safe_load_joblib, safe_load_torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from modules.gat import GATModel
from train_gnn import build_host_graph

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(HERE, "NF-UNSW-NB15-v3.csv")
MODELS_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)


FEATURE_COLS = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]


def load_everything():

    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset rows: {len(df):,}")

    # --------------------------------------------------
    # LABEL
    # --------------------------------------------------

    df["label_enc"] = np.where(
        df["Attack"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "benign",
        0,
        1,
    ).astype(np.int8)

    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------

    X = df[FEATURE_COLS].copy()

    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    X = (
        X
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )

    y = df["label_enc"].values

    src_ips = df["IPV4_SRC_ADDR"].astype(str).values

    # --------------------------------------------------
    # MODELS
    # --------------------------------------------------

    print("Loading RF...")

    rf = safe_load_joblib(
        os.path.join(
            MODELS_DIR,
            "rf_model.pkl"
        )
    )

    print("Loading scaler...")

    scaler = safe_load_joblib(
        os.path.join(
            MODELS_DIR,
            "scaler.pkl"
        )
    )

    print("Loading Autoencoder...")

    # The current environment has a working AE pickle,
    # so prefer it over TensorFlow.
    ae_pkl = os.path.join(
        MODELS_DIR,
        "ae_model.pkl"
    )

    if os.path.exists(ae_pkl):

        ae = safe_load_joblib(ae_pkl)

        print("AE loaded from ae_model.pkl")

    else:

        raise FileNotFoundError(
            "models/ae_model.pkl not found."
        )

    # --------------------------------------------------
    # AE PARAMETERS
    # --------------------------------------------------

    with open(
        os.path.join(
            MODELS_DIR,
            "artifacts.json"
        ),
        "r"
    ) as f:

        artifacts = json.load(f)

    ae_mse_max = float(
        artifacts.get(
            "ae_mse_max_train",
            1.0
        )
    )

    ae_threshold = float(
        artifacts.get(
            "ae_threshold",
            0.1
        )
    )

    # --------------------------------------------------
    # FUSION
    # --------------------------------------------------

    print("Loading Fusion model...")

    fusion = safe_load_joblib(
        os.path.join(
            MODELS_DIR,
            "fusion_model.pkl"
        )
    )

    # --------------------------------------------------
    # GAT
    # --------------------------------------------------

    print("Loading GAT...")

    with open(
        os.path.join(
            MODELS_DIR,
            "gat_artifacts.json"
        ),
        "r"
    ) as f:

        gat_art = json.load(f)

    gat = GATModel(
        input_dim=gat_art["input_dim"],
        hidden_dim=gat_art["hidden_dim"],
        output_dim=2,
    )

    gat.load_state_dict(
        safe_load_torch(
            os.path.join(
                MODELS_DIR,
                "gat_model.pt"
            ),
            map_location="cpu",
            weights_only=True,
        )
    )

    gat.eval()

    graph_scaler = safe_load_joblib(
        os.path.join(
            MODELS_DIR,
            "graph_feature_scaler.pkl"
        )
    )

    # --------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------

    print("Building host graph for GAT...")

    nodes, feats, labels, edge_index = build_host_graph(df)

    feats_scaled = graph_scaler.transform(
        feats
    ).astype(np.float32)

    from torch_geometric.data import Data

    graph_data = Data(
        x=torch.tensor(feats_scaled),
        edge_index=edge_index,
    )

    with torch.no_grad():

        gat_output = gat(graph_data)

        gat_probs = F.softmax(
            gat_output,
            dim=1
        )[:, 1].numpy()

    host_gat_prob = dict(
        zip(
            nodes,
            gat_probs
        )
    )

    default_gat_prob = float(
        np.median(gat_probs)
    )

    return (
        df,
        X,
        y,
        src_ips,
        rf,
        scaler,
        ae,
        ae_mse_max,
        ae_threshold,
        fusion,
        host_gat_prob,
        default_gat_prob,
    )


def evaluate_model(
    name,
    y_true,
    predictions,
    probabilities,
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    cm = confusion_matrix(
        y_true,
        predictions
    ).tolist()

    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
    }

    print(
        f"{name:10s} | "
        f"Acc: {accuracy:.6f} | "
        f"Prec: {precision:.6f} | "
        f"Recall: {recall:.6f} | "
        f"F1: {f1:.6f} | "
        f"AUC: {auc:.6f}"
    )

    print(
        f"             CM: {cm}"
    )

    return result


def main():

    print("=" * 70)
    print("CYBERGRAPH END-TO-END EVALUATION")
    print("=" * 70)

    (
        df,
        X,
        y,
        src_ips,
        rf,
        scaler,
        ae,
        ae_mse_max,
        ae_threshold,
        fusion,
        host_gat_prob,
        default_gat_prob,
    ) = load_everything()

    # --------------------------------------------------
    # SAME HOLDOUT STRATEGY
    # --------------------------------------------------

    (
        _,
        X_test,
        _,
        y_test,
        _,
        src_test,
    ) = train_test_split(
        X,
        y,
        src_ips,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print()
    print(
        f"End-to-end evaluation flows: "
        f"{len(X_test):,}"
    )

    print(
        f"Normal flows: "
        f"{np.sum(y_test == 0):,}"
    )

    print(
        f"Attack flows: "
        f"{np.sum(y_test == 1):,}"
    )

    # --------------------------------------------------
    # RF
    # --------------------------------------------------

    print()
    print("-" * 70)
    print("SCORING BASE MODELS")
    print("-" * 70)

    rf_class_index = int(
        np.where(
            rf.classes_ == 1
        )[0][0]
    )

    rf_probs = rf.predict_proba(
        X_test
    )[:, rf_class_index]

    rf_preds = (
        rf_probs >= 0.5
    ).astype(np.int8)

    # --------------------------------------------------
    # AE
    # --------------------------------------------------

    X_test_scaled = scaler.transform(
        X_test
    ).astype(np.float32)

    # Works with sklearn MLPRegressor
    # and avoids TensorFlow dependency.
    recon = ae.predict(
        X_test_scaled
    )

    ae_mse = np.mean(
        np.square(
            recon - X_test_scaled
        ),
        axis=1,
    )

    ae_probs = np.minimum(
        ae_mse / (ae_mse_max + 1e-9),
        1.0,
    )

    ae_preds = (
        ae_mse > ae_threshold
    ).astype(np.int8)

    # --------------------------------------------------
    # GAT
    # --------------------------------------------------

    gat_probs = np.array(
        [
            host_gat_prob.get(
                ip,
                default_gat_prob
            )
            for ip in src_test
        ]
    )

    gat_preds = (
        gat_probs >= 0.5
    ).astype(np.int8)

    # --------------------------------------------------
    # FUSION
    # --------------------------------------------------

    fusion_features = np.column_stack(
        [
            rf_probs,
            np.minimum(
                ae_mse / (ae_mse_max + 1e-9),
                1.0,
            ),
            gat_probs,
        ]
    )

    fusion_probs = fusion.predict_proba(
        fusion_features
    )[:, 1]

    fusion_preds = (
        fusion_probs >= 0.5
    ).astype(np.int8)

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    results = {}

    results["RF"] = evaluate_model(
        "RF",
        y_test,
        rf_preds,
        rf_probs,
    )

    results["AE"] = evaluate_model(
        "AE",
        y_test,
        ae_preds,
        ae_probs,
    )

    results["GAT"] = evaluate_model(
        "GAT",
        y_test,
        gat_preds,
        gat_probs,
    )

    results["Fusion"] = evaluate_model(
        "Fusion",
        y_test,
        fusion_preds,
        fusion_probs,
    )

    # --------------------------------------------------
    # FUSION COEFFICIENTS
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("LEARNED FUSION PARAMETERS")
    print("=" * 70)

    print(
        "Features: "
        "[RF probability, AE normalized error, GAT probability]"
    )

    print(
        "Coefficients:",
        fusion.coef_[0]
    )

    print(
        "Intercept:",
        fusion.intercept_[0]
    )

    # --------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------

    report = {
        "evaluation": {
            "dataset": "NF-UNSW-NB15-v3",
            "test_size": len(y_test),
            "normal_flows": int(
                np.sum(y_test == 0)
            ),
            "attack_flows": int(
                np.sum(y_test == 1)
            ),
        },

        "models": results,

        "fusion": {
            "features": [
                "rf_probability",
                "ae_normalized_error",
                "gat_probability",
            ],
            "coefficients": fusion.coef_[0].tolist(),
            "intercept": float(
                fusion.intercept_[0]
            ),
        },
    }

    output_path = os.path.join(
        OUT_DIR,
        "end_to_end_evaluation.json"
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Report saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()