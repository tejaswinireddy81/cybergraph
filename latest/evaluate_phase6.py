"""
evaluate_phase6.py
===================
PHASE 6 MASTER EVALUATOR: Attack-Path Prediction & Explainable Graph AI (XAI)

Evaluates:
  1. AttackPathPredictor: Predicts multi-hop attack propagation paths in strict alternating bipartite topology:
     HOST A -> SERVICE X -> HOST B -> SERVICE Y -> HOST C
  2. ExplainableGraphAI: Computes Standardized Feature Saliency Attributions and Temporal Representation Vector Shifts.

Saves reports to:
  - outputs/phase6_explanation_report.json
  - outputs/attack_path_analysis.txt
"""

import os
import json
import numpy as np
import joblib
import torch

from modules.temporal_graph_storage import TemporalGraphStorage
from modules.dynamic_temporal_gnn import DynamicBipartiteTemporalGNN
from modules.attack_path_predictor import AttackPathPredictor
from modules.explainable_graph_ai import ExplainableGraphAI

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TEMPORAL_DATASET_PATH = os.path.join(MODELS_DIR, "temporal_graph_dataset.pt")
GNN_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_temporal_gnn.pt")
SCALERS_PATH = os.path.join(MODELS_DIR, "gnn_scalers.pkl")
PHASE6_JSON_PATH = os.path.join(OUT_DIR, "phase6_explanation_report.json")
PHASE6_TXT_PATH = os.path.join(OUT_DIR, "attack_path_analysis.txt")


def log(msg=""):
    print(msg, flush=True)


def evaluate_phase6():
    log("==================================================")
    log("PHASE 6: ATTACK-PATH PREDICTION & EXPLAINABLE GRAPH AI (XAI)")
    log("==================================================")

    if not (os.path.exists(TEMPORAL_DATASET_PATH) and os.path.exists(GNN_MODEL_PATH) and os.path.exists(SCALERS_PATH)):
        raise FileNotFoundError("Prerequisite Phase 5 model artifacts missing. Run train_temporal_gnn.py first.")

    storage = TemporalGraphStorage.load(TEMPORAL_DATASET_PATH)
    scalers = joblib.load(SCALERS_PATH)
    host_scaler = scalers["host_scaler"]
    service_scaler = scalers["service_scaler"]
    edge_scaler = scalers["edge_scaler"]

    sample_snap = storage.snapshots[0]
    gnn_model = DynamicBipartiteTemporalGNN(
        host_dim=sample_snap.host_x.shape[1],
        service_dim=sample_snap.service_x.shape[1],
        edge_dim=sample_snap.edge_attr.shape[1],
        hidden_dim=32,
    )
    gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH))
    gnn_model.eval()

    train_split_idx = int(round(len(storage.snapshots) * 0.75))
    train_snaps = storage.snapshots[:train_split_idx]
    test_snaps = storage.snapshots[train_split_idx:]

    log(f"Loaded {len(storage.snapshots)} snapshots: {len(train_snaps)} Train, {len(test_snaps)} Test.")

    # 1. Train Attack-Path Predictor on Training Snapshots
    predictor = AttackPathPredictor(risk_threshold=0.5)
    predictor.fit_transitions(train_snaps)
    log("Attack-Path Predictor fitted on historical temporal snapshot transitions.")

    # 2. Instantiate Explainable Graph AI Engine
    xai = ExplainableGraphAI()

    # 3. Propagate GNN across train snapshots to warm up temporal memory states
    prev_h, prev_s = None, None
    with torch.no_grad():
        for snap in train_snaps:
            if snap.edge_index_host_service.shape[1] == 0:
                continue
            hx_scaled = torch.tensor(host_scaler.transform(snap.host_x), dtype=torch.float32)
            sx_scaled = torch.tensor(service_scaler.transform(snap.service_x), dtype=torch.float32)
            eidx = torch.tensor(snap.edge_index_host_service, dtype=torch.long)
            eattr_scaled = torch.tensor(edge_scaler.transform(snap.edge_attr), dtype=torch.float32)
            _, prev_h, prev_s = gnn_model(hx_scaled, sx_scaled, eidx, eattr_scaled, prev_h, prev_s)

        # 4. Predict Attack Paths and Generate XAI Explanations on Test Snapshots (G_6, G_7)
        all_attack_paths = []
        all_xai_explanations = []

        for snap in test_snaps:
            if snap.edge_index_host_service.shape[1] == 0:
                continue

            hx_scaled = torch.tensor(host_scaler.transform(snap.host_x), dtype=torch.float32)
            sx_scaled = torch.tensor(service_scaler.transform(snap.service_x), dtype=torch.float32)
            eidx = torch.tensor(snap.edge_index_host_service, dtype=torch.long)
            eattr_scaled = torch.tensor(edge_scaler.transform(snap.edge_attr), dtype=torch.float32)

            old_h = prev_h.numpy() if prev_h is not None else None

            logits, prev_h, prev_s = gnn_model(hx_scaled, sx_scaled, eidx, eattr_scaled, prev_h, prev_s)
            probs = torch.sigmoid(logits).numpy()

            curr_h = prev_h.numpy()

            # Attack Path Prediction
            paths = predictor.predict_attack_paths(snap, gnn_probs=probs, top_k_paths=3)
            all_attack_paths.extend(paths)

            # XAI Explanations
            exps = xai.explain_snapshot(
                snapshot=snap,
                gnn_probs=probs,
                host_states=curr_h,
                prev_host_states=old_h,
                top_k=3,
            )
            all_xai_explanations.extend(exps)

    report_json = {
        "phase": "Phase 6: Attack-Path Prediction & Explainable Graph AI",
        "snapshots_evaluated": len(test_snaps),
        "predicted_attack_paths": all_attack_paths,
        "xai_flow_explanations": all_xai_explanations,
    }

    with open(PHASE6_JSON_PATH, "w") as f:
        json.dump(report_json, f, indent=2)

    with open(PHASE6_TXT_PATH, "w") as f:
        f.write("==================================================\n")
        f.write("PHASE 6: ATTACK-PATH PREDICTION & XAI ANALYSIS REPORT\n")
        f.write("==================================================\n\n")

        f.write("--- PREDICTED MULTI-HOP ATTACK PROPAGATION PATHS (STRICT BIPARTITE TOPOLOGY) ---\n")
        for i, path in enumerate(all_attack_paths, 1):
            f.write(f"Path #{i}:\n")
            f.write(f"  Source Host           : {path['source_host']}\n")
            f.write(f"  Primary Attack Target : {path['primary_attack_target']}\n")
            f.write(f"  Risk Score            : {path['attack_risk_score']:.4f}\n")
            f.write(f"  Predicted Next Target : {path['predicted_next_target']}\n")
            f.write(f"  Confidence            : {path['prediction_confidence']*100:.1f}%\n")
            f.write(f"  Bipartite Sequence    : {' -> '.join(path['bipartite_attack_chain'])}\n\n")

        f.write("--- MODEL-DRIVEN XAI FLOW EXPLANATIONS (STANDARDIZED FEATURE SALIENCY ATTRIBUTION) ---\n")
        for i, exp in enumerate(all_xai_explanations, 1):
            f.write(f"Explanation #{i}:\n")
            f.write(f"  Flow                  : {exp['source_host']} -> {exp['target_service']}\n")
            f.write(f"  Risk Score            : {exp['overall_risk_score']:.4f}\n")
            f.write(f"  Primary Saliency      : {exp['primary_saliency_driver']}\n")
            f.write(f"  Attributions          : {exp['feature_attributions']}\n")
            f.write(f"  Methodology           : {exp['attribution_methodology']}\n")
            f.write(f"  Summary               : {exp['explanation_summary']}\n\n")

    log("\n==================================================")
    log("PHASE 6 ANALYSIS COMPLETED SUCCESSFULLY")
    log("==================================================")
    log(f"Generated {len(all_attack_paths)} predicted attack propagation paths.")
    log(f"Generated {len(all_xai_explanations)} model-driven XAI flow explanations.")
    log(f"Saved reports to '{PHASE6_JSON_PATH}' and '{PHASE6_TXT_PATH}'.")

    return report_json


if __name__ == "__main__":
    evaluate_phase6()
