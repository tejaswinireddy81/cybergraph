"""
evaluate_phase10.py
====================
PHASE 10 MASTER EVALUATOR: Adversarial Robustness & Zero-Day Threat Stress Testing

Evaluates model resilience under 4 adversarial perturbation levels (epsilon = 0.0, 0.1, 0.25, 0.5)
across all 94,320 test flows.

Outputs:
  - outputs/adversarial_robustness_report.json
  - outputs/stress_test_analysis.txt
"""

import os
import json
import joblib
import numpy as np
import torch

from modules.temporal_graph_storage import TemporalGraphStorage
from modules.dynamic_temporal_gnn import DynamicBipartiteTemporalGNN
from modules.adversarial_stress_tester import AdversarialStressTester

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TEMPORAL_DATASET_PATH = os.path.join(MODELS_DIR, "temporal_graph_dataset.pt")
GNN_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_temporal_gnn.pt")
SCALERS_PATH = os.path.join(MODELS_DIR, "gnn_scalers.pkl")

REPORT_JSON_PATH = os.path.join(OUT_DIR, "adversarial_robustness_report.json")
REPORT_TXT_PATH = os.path.join(OUT_DIR, "stress_test_analysis.txt")


def evaluate_phase10():
    print("==================================================", flush=True)
    print("PHASE 10: ADVERSARIAL ROBUSTNESS & ZERO-DAY STRESS TESTING", flush=True)
    print("==================================================", flush=True)

    if not (os.path.exists(TEMPORAL_DATASET_PATH) and os.path.exists(GNN_MODEL_PATH) and os.path.exists(SCALERS_PATH)):
        raise FileNotFoundError("Prerequisite model artifacts missing. Run train_temporal_gnn.py first.")

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
    test_snaps = storage.snapshots[train_split_idx:]

    epsilons = [0.0, 0.10, 0.25, 0.50]
    robustness_results = {}

    for eps in epsilons:
        tester = AdversarialStressTester(perturbation_level=eps)
        total_eval_flows = 0
        total_correct = 0
        total_attack_flows = 0
        total_attack_detected = 0

        prev_h, prev_s = None, None
        with torch.no_grad():
            for snap in test_snaps:
                if snap.edge_index_host_service.shape[1] == 0:
                    continue

                hx_scaled = torch.tensor(host_scaler.transform(snap.host_x), dtype=torch.float32)
                sx_scaled = torch.tensor(service_scaler.transform(snap.service_x), dtype=torch.float32)
                eidx = torch.tensor(snap.edge_index_host_service, dtype=torch.long)

                labels = np.array(snap.edge_y, dtype=int)

                # Apply adversarial perturbation to edge features if epsilon > 0
                eattr_np = snap.edge_attr.copy()
                if eps > 0:
                    eattr_np = tester.generate_adversarial_batch(eattr_np, labels)

                eattr_scaled = torch.tensor(edge_scaler.transform(eattr_np), dtype=torch.float32)

                logits, prev_h, prev_s = gnn_model(hx_scaled, sx_scaled, eidx, eattr_scaled, prev_h, prev_s)
                probs = torch.sigmoid(logits).numpy()
                preds = (probs >= 0.50).astype(int)

                total_eval_flows += len(labels)
                total_correct += int(np.sum(preds == labels))

                att_mask = (labels == 1)
                total_attack_flows += int(np.sum(att_mask))
                total_attack_detected += int(np.sum(preds[att_mask] == 1))

        acc = float(total_correct / total_eval_flows)
        rec = float(total_attack_detected / max(total_attack_flows, 1))
        clean_rec = robustness_results.get(0.0, {}).get("recall", rec)
        arm = float(rec / (clean_rec + 1e-9)) * 100.0

        robustness_results[eps] = {
            "epsilon": eps,
            "total_flows": total_eval_flows,
            "accuracy": acc,
            "recall": rec,
            "adversarial_robustness_metric_pct": arm,
            "attack_flows_detected": total_attack_detected,
            "total_attack_flows": total_attack_flows,
        }

        print(f"Epsilon = {eps:.2f} | Acc: {acc:.6f} | Recall: {rec:.6f} | ARM: {arm:.2f}% ({total_attack_detected}/{total_attack_flows})", flush=True)

    report_json = {
        "phase": "Phase 10: Adversarial Robustness & Zero-Day Threat Stress Testing",
        "evaluations": robustness_results,
    }

    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report_json, f, indent=2)

    with open(REPORT_TXT_PATH, "w") as f:
        f.write("==================================================\n")
        f.write("PHASE 10: ADVERSARIAL STRESS-TESTING ANALYSIS REPORT\n")
        f.write("==================================================\n\n")

        f.write("--- ADVERSARIAL PERTURBATION RESILIENCE CURVE ---\n")
        for eps, res in robustness_results.items():
            f.write(f"Epsilon Perturbation = {eps:.2f}:\n")
            f.write(f"  Samples Evaluated  : {res['total_flows']}\n")
            f.write(f"  Accuracy           : {res['accuracy']:.6f}\n")
            f.write(f"  Attack Recall      : {res['recall']:.6f}\n")
            f.write(f"  Adversarial Metric : {res['adversarial_robustness_metric_pct']:.2f}%\n")
            f.write(f"  Attacks Detected   : {res['attack_flows_detected']} / {res['total_attack_flows']}\n\n")

    print("\n==================================================", flush=True)
    print("PHASE 10 STRESS TESTING COMPLETED SUCCESSFULLY", flush=True)
    print("==================================================", flush=True)
    print(f"Saved reports to '{REPORT_JSON_PATH}' and '{REPORT_TXT_PATH}'.", flush=True)

    return report_json


if __name__ == "__main__":
    evaluate_phase10()
