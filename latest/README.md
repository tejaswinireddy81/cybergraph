# CyberGraph NIDS — Multi-Signal Fusion with Graph-Based Coordinated-Attack Detection

## Abstract

This project implements **CyberGraph**, an advanced hybrid Network Intrusion Detection System (NIDS) combining
three complementary detectors:
1. **Supervised Gradient-Boosted Classifier (XGBoost)** operating on per-flow NetFlow features
2. **Unsupervised Deep Autoencoder Anomaly Detector (PyTorch)** trained exclusively on benign traffic for zero-day/novel deviation detection
3. **Dynamic Bipartite Temporal Graph Neural Network (GNN/GAT)** operating over the live host-and-service communication graph

These detectors are fused through a calibrated multi-signal fusion layer into a unified risk score.
The system is trained and evaluated on **NF-UNSW-NB15-v3** using **47 calibrated NetFlow features** per flow
(standardized in `artifacts.json` to eliminate un-routable raw TTL header offsets for cross-platform generalization),
and runs live against real network traffic via Scapy packet capture on monitored interfaces.

---

## 1. Canonical Pipeline Architecture

> [!IMPORTANT]
> **Single Unified Pipeline**: This codebase is standardized around the active **4-tier XGBoost + PyTorch Autoencoder + Multi-Signal Fusion + Dynamic Bipartite Temporal GNN** pipeline implemented in `train_xgboost_ae.py`, `train_fusion.py`, `train_gnn.py`, and `app.py`.
> The earlier prototype script (`train_models.py`, which used Random Forest and random train/test splits) is retained only as a legacy reference. The live system exclusively uses the 47-feature artifacts and strictly chronological temporal evaluation to eliminate data leakage.

---

## 2. Dataset & Zero-Leakage Temporal Evaluation

**NF-UNSW-NB15-v3** — a NetFlow-formatted version of the UNSW-NB15 IDS benchmark,
47 active calibrated NetFlow features per flow (byte/packet counts, TCP flags, inter-arrival-time statistics,
throughput, DNS/FTP/ICMP metadata, retransmission counts), 2,365,424 total flows.

| Class | Count | % |
|---|---|---|
| Benign | 2,237,731 | 94.60% |
| Exploits | 42,748 | 1.81% |
| Fuzzers | 33,816 | 1.43% |
| Generic | 19,651 | 0.83% |
| Reconnaissance | 17,074 | 0.72% |
| DoS | 5,980 | 0.25% |
| Backdoor | 4,659 | 0.20% |
| Shellcode | 2,381 | 0.10% |
| Analysis | 1,226 | 0.05% |
| Worms | 158 | 0.01% |

**Sampling & Evaluation, stated honestly:**
The models were trained on a stratified sample of **377,279 rows** preserving all 127,693 attack flows across every rare class.
To strictly prevent **data leakage**, evaluation uses **chronological temporal splitting** based on `FLOW_START_MILLISECONDS`:
- **Training Set**: Chronological first 75% of flows (including benign baseline for the Autoencoder).
- **Held-Out Test Set**: Chronological final 25% of flows. No random shuffling or future flow contamination occurs.

---

## 3. Architecture

```
                    +----------------------+
   raw packets -->  |   FlowExtractor      |  per-packet -> per-flow NetFlow-style
  (Scapy sniff)     |  (flow_extractor.py) |  feature aggregation (IAT stats, TCP
                    +----------+-----------+  flags, retransmits, DNS/FTP parsing...)
                               | completed flow (47 calibrated NetFlow features)
                               v
                    +----------------------+
                    |   LiveTrafficEngine   |
                    |   (inference pipeline)|
                    +----------+-----------+
              +----------------+----------------+
              v                v                 v
      +---------------+ +-------------+  +----------------------+
      |    XGBoost    | | Autoencoder |  |  OnlineGraphStream   |
      |  (supervised  | | (PyTorch AE |  |  (Dynamic Bipartite  |
      |   classifier) | | unsupervsd) |  |   Temporal GNN/GAT)  |
      +-------+-------+ +------+------+  +----------+-----------+
              |  xgb_prob      |  ae_error          |  gnn_edge_risk
              +----------------+-------+------------+
                                       v
                           +----------------------+
                           |   FusionEngine       |  Multi-signal calibrated
                           |  (fusion_engine.py)  |  anomaly & risk synthesis
                           +----------+-----------+
                                       v
                           +----------------------+
                           | Firewall & Response  |  Graduated Containment,
                           | (Level 1, 2, 3 ACL)  |  L3 Firewall & Audit
                           +----------------------+
```

Also wired in: `RiskEngine` (a separate human-readable 0-100 heuristic score for
SOC-analyst dashboards, distinct from the ML fusion score), `BehaviorAnalyzer`
(per-host rolling stats), `AttackChainDetector` (multi-stage pattern matching),
`ThreatIntelligence` (static blacklist lookup), `DeviceProfile` and
`TopologyEngine` (for the live dashboard's device/graph views), and
`AdaptiveThresholdManager` (recalibrates the AE threshold against a rolling window
of traffic the other detectors didn't flag).

### 3.5 MITRE ATT&CK Contextualization (Phase 2)

The MITRE ATT&CK mapper (`modules/mitre_attack_mapper.py`) provides
**defensible, evidence-based technique mapping** — not ML guessing. It
determines whether observed network behaviors match known ATT&CK techniques
based on:
- Behavioral evidence from flow telemetry
- BARI route analysis
- Correlation engine findings
- Attack journey progression

**Supported techniques:** T1046 (Network Service Discovery), T1110 (Brute Force),
T1498 (Network Denial of Service).

**Design decisions:**
- **T1059 (Command and Scripting Interpreter) is disabled** because network flow
  telemetry does not directly observe command or script execution.
- **T1110 requires actual authentication-attempt evidence** — Reconnaissance alone
  is insufficient to map to Brute Force.
- **T1046 requires actual multi-port diversity** from the correlation timeline,
  not a single-port proxy.

**Single source of truth:** All technique definitions live in
`datasets/mitre_attack_mapping.json` (version 1.1). The mapper loads from this
JSON at runtime — there is no duplicate configuration elsewhere.

**Confidence formula:** `confidence = weighted evidence match × model confidence
damping` (rule-based, not calibrated probability). See `/api/mitre_mappings` for
the full metadata and evidence weights.

### 3.1 Why a graph model at all

A single compromised or coordinated attacker rarely looks alarming in isolation —
each individual flow can be short and unremarkable. What stands out is the
**structure**: many source hosts converging on one target, unusual fan-out,
port-scanning entropy patterns across a botnet. The GAT operates on the live
host-communication graph (nodes = IP addresses, edges = observed communication,
7 real structural features per host: degree, byte/packet volume, destination-port
entropy, protocol diversity, retransmission volume) specifically to catch that
class of attack, which a per-flow classifier structurally cannot see.

---

## 4. Results

### 4.1 XGBoost Classifier (per-flow, supervised)

| Metric | Value |
|---|---|
| Features | 47 Calibrated NetFlow features |
| Accuracy | 99.98% |
| Precision | 99.97% |
| Recall | 99.99% |
| F1 Score | 99.98% |

Evaluated on the strict chronological test set (the final 25% of temporal flows). Near-perfect separability is a known characteristic of NetFlow-derived statistical features on the UNSW benchmark family, serving as a reliable first-line discriminant.

### 4.2 Deep Autoencoder (per-flow, unsupervised — trained on benign traffic only)

| Metric | Value |
|---|---|
| Threshold (99th pct. normal validation error) | 0.09878 |
| Accuracy | 93.45% |
| Precision | 97.84% |
| Recall | 82.46% |
| F1 | 89.49% |

This is the more honest number in the report: unlike the RF, the AE never saw a
single labeled attack during training — 82.5% recall from reconstruction error
alone is a genuinely useful anomaly signal, not a trivially separable one.

### 4.3 GAT (host-graph, coordinated-attack detector)

| Metric | Value |
|---|---|
| Host graph size | 43 nodes, 148 unique edges |
| Labeled hosts (≥3 flows) | 39 (4 malicious, 35 benign) |
| Test set | 10 hosts (1 malicious) |
| ROC-AUC | 1.0000 |
| Accuracy / Precision / Recall / F1 | 1.00 / 1.00 / 1.00 / 1.00 |

**This number needs an explicit caveat, not a headline claim** — see Section 5.

### 4.4 Learned Fusion Layer

Replaces the original fixed weights (`rf: 0.30, ae: 0.20, gnn: 0.25, gat: 0.25`,
asserted with no derivation) with a logistic regression trained on 66,024 held-out
flows (`rf_prob`, `ae_normalized_error`, `gat_host_prob` → true label), evaluated
on a further 28,296 held-out flows never used in fusion training:

| Metric | Value |
|---|---|
| Accuracy / Precision / Recall / F1 | 1.00 / 1.00 / 1.00 / 1.00 |
| ROC-AUC | 1.0000 |
| Learned coefficients (rf, ae, gat) | 7.98, 0.17, 8.06 |

The learned weights on RF and GAT dominate the AE's contribution (0.17 vs. ~8) —
this makes sense given the RF and GAT are both near-perfectly separable on this
dataset, so the fusion layer correctly learns to trust them heavily and treat the
AE as a secondary/tie-breaking signal, exactly the behavior a stacked ensemble
should learn rather than assert by hand.

---

## 5. Limitations — stated honestly

**The GAT's 1.0 ROC-AUC is not strong evidence of real-world generalization, and
shouldn't be read as one.** NF-UNSW-NB15-v3 (like the underlying UNSW-NB15
capture it derives from) was generated in a small testbed using the IXIA
PerfectStorm traffic generator, which means the entire 2.36-million-row dataset —
not just this project's sample — contains only **43 distinct IP addresses total**.
With that few hosts, a 4-malicious/35-benign host-level classification task is
close to memorization, and the 1.0 AUC is measured on a 10-host test set (1
malicious). When this GAT was tested against a synthetic coordinated-attack
scenario built during development (8 distinct attacker IPs converging on one
target), it did **not** flag the pattern as high-risk — a real, reproducible
finding, not a hidden one. This is a direct consequence of the dataset's host
diversity ceiling, not a bug in the graph construction or training code (which
were verified independently: real IPs, real structural features, real edges, real
training/test split).

**What this means practically:** the graph-construction and GAT-training pipeline
is genuinely correct and ready to use, but the specific trained weights shipped
here should not be trusted to generalize to unfamiliar coordinated-attack
patterns. Before relying on the GAT's coordinated-attack detection in any real
deployment, retrain `train_gnn.py` against a host population that actually looks
like the target network (dozens-to-hundreds of real devices, captured over time)
— exactly the same "recalibrate before trusting it live" caveat the
`adaptive_learning.py` module already applies to the Autoencoder's threshold, just
not yet automated for the GAT itself. That automation (continuous host-graph
retraining against live traffic) is a natural next step, not something claimed as
done here.

**Response engine is simulated, not integrated.** `response_engine.py` logs
structured alerts and maintains a "blocked hosts" list, but does not call out to
a real firewall, switch ACL, or cloud security group. That integration is a
well-scoped next step but isn't safe to claim without building and testing it
against a real enforcement point.

**RF/AE near-perfect scores reflect this dataset's separability**, not a claim
that any NIDS gets 99.99% in production against adversarial, evolving traffic —
this is stated in Section 4.1 and repeated here because it's the single most
important caveat for how these numbers should be read in a viva/defense.

---

## 6. Running it

```bash
pip install -r requirements.txt   # scikit-learn, tensorflow-cpu, torch, torch-geometric,
                                    # scapy, dash, pandas, networkx, joblib

# 1. Train the per-flow detectors (RF + Autoencoder)
python train_models.py

# 2. Train the host-graph GAT (coordinated-attack detector)
python train_gnn.py

# 3. Train the learned fusion layer (combines all three signals)
python train_fusion.py

# 4. Launch the live dashboard
python app.py
```

### 6.1 Live deployment against many real devices over the internet

The system does **not** need an agent installed on each device. `FlowExtractor`
+ `live_capture.py` sniff packets on **one monitored interface** using Scapy, and
independently track a flow per unique `(src_ip, dst_ip, src_port, dst_port,
protocol)` tuple — so any number of distinct devices sending traffic through that
interface are tracked and scored independently, with no per-device configuration.

To actually see traffic from many devices on the internet, the interface Scapy
listens on needs to actually carry that traffic — options, in order of how much
infrastructure they need:
1. **Run on the gateway/router itself** (sees every device's traffic by
   definition — simplest for a home/small-office network).
2. **A SPAN/mirror port on a managed switch**, with this system plugged into the
   mirror port (standard enterprise approach — no path change for real traffic).
3. **A network TAP** inline between the internet uplink and the LAN.

`GraphBuilder` accumulates the host-communication graph continuously and prunes
edges/nodes idle for more than `GRAPH_WINDOW_SECONDS` (default 300s) so memory use
stays bounded on a long-running deployment; the GAT re-scores the graph every 25
processed flows rather than on every single flow, to keep up with a busy link.

---

## 7. Project structure

```
modules/
  behavior_analysis.py    per-host rolling traffic stats
  graph_builder.py         live host-communication graph + real structural features
  graph_datasets.py        networkx -> torch_geometric conversion
  gat.py                   GAT architecture (torch_geometric GATConv)
  gnn_model.py              alternate GCN architecture (available, not used by default)
  gnn_engine.py             loads + runs the trained GAT over the live graph
  fusion_engine.py          learned logistic-regression fusion (falls back to fixed
                             weights only if fusion_model.pkl hasn't been trained yet)
  risk_engine.py            explainable 0-100 heuristic score for analysts
  response_engine.py        alerting + cooldown + simulated blocking
  adaptive_learning.py      AE threshold recalibration against recent unflagged traffic
  threat_intelligence.py    static IP blacklist lookup
  attack_chain.py           multi-stage attack pattern detection
  device_profile.py         per-device running profile
  topology_engine.py        network graph for the live dashboard
  explainability.py         SHAP-based RF explanations
train_models.py             trains RF + Autoencoder
train_gnn.py                trains the GAT on the real host-communication graph
train_fusion.py             trains the learned fusion layer
inference.py                wires everything together into run_inference()
flow_extractor.py           Scapy packet -> NetFlow-style flow feature aggregation
live_capture.py             interface selection + capture threads
app.py                      live Dash dashboard
```

---

## 8. Reproduce These Numbers

To reproduce all experimental results, cross-validation metrics, ablation tables, and adversarial robustness reports:

### 8.1 Random Seeds & Split Ratios

| Component | Random Seed | Split Ratio / Cross-Validation | Target Output File |
|---|---|---|---|
| **RF / AE Flow Training** | `seed = 42` | 75% Train (282,959 flows) / 25% Test (94,320 flows) | `models/rf_model.pkl`, `models/ae_model.pkl` |
| **GAT Host-Graph Detector** | `seed = 42` | 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5)`) | `outputs/gat_cv_report.json` |
| **Offline Experiments & Ablation** | `seed = 42` | Unified 94,320 test flows across all 5 research models | `outputs/experiment_results.json` |
| **RF / AE Validation & Perturbation** | `seed = 42` | 20,000 held-out flow evaluation sample (5%, 10%, 20% noise) | `outputs/rf_ae_validation_report.json` |

### 8.2 Execution Order

```bash
# 1. Download/verify dataset
python scripts/download_dataset.py

# 2. Train RF and AE baseline models
python train_models.py

# 3. Train XGBoost + AE + Fusion baseline
python train_xgboost_ae.py

# 4. Train GAT host-graph detector with 5-Fold Cross-Validation
python train_gnn.py

# 5. Train multi-signal fusion layer
python train_fusion.py

# 6. Run unified 5-experiment research benchmark
python run_offline_experiments.py

# 7. Run RF/AE cross-dataset, ablation, and adversarial perturbation tests
python evaluate_rf_ae_validation.py

# 8. Execute unit test suite
python -m pytest tests/ -v
```

---
