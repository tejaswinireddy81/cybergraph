# Legacy — CyberGraph Generation 1 Pipeline

This directory preserves the historical **Generation 1 (RF + Keras AE + Static GAT)** implementation
for project traceability. These files are **NOT active** in the canonical production pipeline.

---

## Why These Files Were Moved

| File | Reason |
|------|--------|
| `rf_pipeline/inference.py` | Orphaned — neither `app.py` nor `live_capture.py` imports it. Uses RF + Keras AE + Static GAT. |
| `rf_pipeline/train_models.py` | Legacy training script for Random Forest + Keras Autoencoder. Superseded by `train_xgboost_ae.py`. |
| `rf_pipeline/train_fusion.py` | Trains 3-feature fusion model `[rf, ae, gat]`. Conflicts with canonical 2-feature `[xgb, ae]` model written to the same `fusion_model.pkl` path. |
| `rf_pipeline/evaluate_rf_ae_validation.py` | Evaluation script for the legacy RF+AE pipeline. No longer relevant. |
| `static_gat/train_gnn.py` | Trains the old static homogeneous host-to-host GAT. Superseded by `train_temporal_gnn.py`. |
| `static_gat/modules/gat.py` | Old static GAT model (2-layer GATConv, node classification). Replaced by `dynamic_temporal_gnn.py`. |
| `static_gat/modules/gat_engine.py` | **Dead code** — exact duplicate of `gnn_engine.py`. Never imported anywhere. |
| `static_gat/modules/gnn_engine.py` | Legacy GAT inference wrapper. Used only by the orphaned `inference.py`. |
| `static_gat/modules/graph_builder.py` | Homogeneous undirected NetworkX host graph. Replaced by `bipartite_graph_builder.py`. |
| `static_gat/modules/graph_dataset.py` | PyG dataset converter for homogeneous host graph. Replaced by bipartite PyG HeteroData. |

---

## Generation 1 vs Generation 2 Architecture

| Dimension | Generation 1 (Legacy) | Generation 2 (Canonical) |
|---|---|---|
| Supervised Classifier | Random Forest (`rf_model.pkl`) | XGBoost (`xgboost_model.json`) |
| Anomaly Detector | Keras AE (`ae_model.keras`) | scikit-learn MLPRegressor AE (`ae_model.pkl`) |
| Graph Topology | Homogeneous host-to-host undirected graph | Heterogeneous bipartite (Host ↔ Service) |
| GNN | Static 2-layer GAT (`gat_model.pt`) | Dynamic Temporal GNN + GRU (`dynamic_temporal_gnn.pt`) |
| GNN Task | Host node classification | Temporal edge/flow classification |
| Fusion | 3-signal logistic regression `[rf, ae, gat]` | 2-signal logistic regression `[xgb, ae]` |
| Feature Schema | 49 NetFlow features (incl. MIN_TTL, MAX_TTL) | 47 NetFlow features (FIX D: TTL removed) |
| Route Intelligence | None | BARI Engine (behavioral route analysis) |
| Journey Analysis | None | Attack Journey Engine |

---

## Historical Performance (Generation 1 Baseline)

These results are from the Generation 1 pipeline on NF-UNSW-NB15-v3:

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Random Forest | 99.9968% | 0.999966 | 1.000000 |
| Autoencoder | 86.7% (approx) | — | — |
| Static GNN (GAT) | 99.9968% | 0.999966 | 1.000000 |

---

## Important Notes

- These files are preserved for **project traceability only**.
- Do NOT run `train_fusion.py` from this directory — it overwrites `models/fusion_model.pkl` with a 3-feature model that will break `LiveTrafficEngine`.
- Do NOT import `inference.py` into the live pipeline — it uses Random Forest and will fail at runtime.
- Model artifacts (`rf_model.pkl`, `ae_model.keras`, `gat_model.pt`, `graph_feature_scaler.pkl`) remain in `models/` for historical reference.

---

*Archived in Phase 1 — Architecture & Pipeline Reconciliation (2026-09-12)*
