# Model improvement & retraining plan

Status: **Revision 3. Infrastructure, dataset verification and the extractor investigation are done; NO full retraining has been run.** No production model,
threshold, firewall rule or Git history was changed. Your decisions from the first review are recorded in section 12; what was
built and learned since is in section 13. A full training run is blocked in code until you approve this revised plan
(`run_candidate.py` refuses to run without `--approved-full-run`).

Inputs: [false_positive_investigation.md](false_positive_investigation.md) and the diagnostics in
`scripts/fp_investigation/` (scripts 01-06).

## 0. Your decisions, and where each is handled

| # | Decision | Where |
|---|---|---|
| 1 | Do not start retraining yet | whole plan is gated; section 10 |
| 2 | Compare feature sets with/without `MIN_IP_PKT_LEN` | section 3 |
| 3 | Investigate the 52-byte TCP floor, compare with/without | section 4 |
| 4 | Benign traffic: DNS, HTTPS, QUIC, TCP, UDP, ARP, IGMP | section 5 (ARP handled separately, see 5.3) |
| 5 | Separate held-out dataset | section 5.4 |
| 6 | Precision, recall, F1, FPR, PR-AUC, ROC-AUC (not accuracy alone) | section 6 |
| 7 | Autoencoder's ~0 fusion weight, recalibration | section 7 |
| 8 | No overwriting; backups; new models kept separate | section 8 |
| 9 | Do not touch existing firewall rules | section 9 |
| 10 | Keep everything uncommitted/unpushed | section 9 |

## 1. What the evidence says (so the plan targets real causes)

1. **Shortcut learning.** One feature (`MIN_IP_PKT_LEN`) holds 79 % of XGBoost importance; alone it gives 0.995 AUC.
   After removing it the model switches to other shortcuts (throughput/rate features) and the benchmark stays at 1.0000 AUC.
2. **The benign class does not look like modern traffic** (script `06_dataset_coverage.py`):
   benign flows in NF-UNSW-NB15-v3 = 249,586, captured over **3 distinct days in Jan-Feb 2015**, and contain
   **0 HTTPS (TCP/443), 0 QUIC, 0 NTP, 0 DHCP, 0 IGMP**, 2 mDNS, 1 SSDP, 28 ICMP flows. DNS (UDP/53) is 16 %, plain HTTP 8 %,
   and 68 % is "TCP other" spread over 42,798 distinct destination ports (generator artifact).
3. **The autoencoder is a good in-distribution detector but fails on live traffic** (script `05_ae_fusion_diagnostic.py`):
   ROC-AUC 0.982 / PR-AUC 0.967 on the UNSW test split, but 99-100 % of live flows exceed its anomaly threshold.
4. **The fusion layer cannot learn from the AE for three separate reasons** (section 7), the first being a methodology flaw:
   it is fit on the **test split**.
5. **ARP and IPv6 never reach the model**: `FlowExtractor` returns early for anything without an IPv4 layer.
6. **The TCP floor** (`max(shortest, 52)`) hides a 30-port SYN scan (0 of 30 flagged vs 30 of 30 without it).

Consequence: retraining the same pipeline on the same data would reproduce these problems. The plan therefore changes the
**features, data, splits, fusion and evaluation together**.

## 2. Objectives and success criteria (proposed, to be approved before any training)

Defined up front so results cannot move the goalposts. Numbers are proposals for you to adjust.

| Criterion | Proposed target (on data never used for training/threshold selection) |
|---|---|
| Live-benign false-positive rate (per flow) | <= 0.5 % overall and <= 1 % for each benign class (DNS, HTTPS, QUIC, TCP, UDP, IGMP) |
| Alerts on an unattended benign soak | <= N critical alerts per hour (you choose N; proposed 2) |
| Recall at that operating point on lab-generated attacks | >= 90 % for SYN scan, port sweep, UDP flood, SYN flood |
| Recall on held-out attack families (leave-one-family-out) | reported per family; target median >= 80 % |
| Concentration | no single feature > 30 % of importance (permutation + gain) |
| Shortcut control | single-feature baselines must not match the full model's AUC by more than 0.02 |
| Honest headline | report PR-AUC/ROC-AUC on *external* and *live* sets, not only on the internal split |

Expected outcome, stated honestly: headline AUC on held-out data will be **lower than the current 1.0000**. That is the
point; the current number overstates real-world quality.

## 3. Feature sets to compare (item 2)

All sets use the same 47-column schema as the starting point so the comparison isolates what is removed.

| ID | Definition | Features | Purpose |
|---|---|---|---|
| **F0** | current production set | 47 | baseline; must reproduce today's behaviour |
| **F1** | F0 minus `MIN_IP_PKT_LEN`, `SHORTEST_FLOW_PKT` | 45 | the direct "with vs without `MIN_IP_PKT_LEN`" answer |
| **F2** | F1 minus packet-size family: `MAX_IP_PKT_LEN`, `LONGEST_FLOW_PKT`, the five `NUM_PKTS_*_BYTES` buckets | 38 | removes the whole size signature |
| **F3** | F2 minus identity-like fields: `L4_SRC_PORT`, `L4_DST_PORT`, `L7_PROTO`, `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER`, `ICMP_TYPE`, `ICMP_IPV4_TYPE`, `FTP_COMMAND_RET_CODE` | 29 | removes generator/identity artifacts |
| **F4** | F3 minus the four rate/throughput columns (`SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT`), minus `PROTOCOL` and the raw `IN/OUT_BYTES/PKTS`, **plus** 12 derived features: `log1p` of in/out bytes, in/out packets and duration, bytes-per-packet (in/out), log out/in byte ratio, TCP/UDP/ICMP/other one-hot | **32** (20 base + 12 derived) | behaviour-only candidate; the rate columns were the next shortcut in the ablation |
| **F5** (optional) | F4 + source/destination port buckets (<1024 / registered / ephemeral) instead of raw ports | **34** (20 base + 14 derived) | keeps useful context without memorising ports |

Notes
* `PROTOCOL` is kept in F0-F3 and replaced by the one-hot class in F4/F5.
* **Feature-semantics audit (new, important).** The dataset was produced by a NetFlow exporter and the live path uses
  scapy `FlowExtractor`. Identical column names may be computed differently (retransmission counters, TCP window,
  IAT units, direction conventions, DNS fields). Before training, each feature is classified as *robust* (counts, bytes,
  duration, flags) or *exporter-dependent*, by reading the extractor and the dataset documentation and by comparing
  distributions on matched benign activity. Exporter-dependent features are dropped or flagged in every set.

Each set is trained with 3 random seeds and identical hyper-parameters (XGBoost n_estimators=150, max_depth=6,
lr=0.1) so differences come from the features, not tuning. A second pass with modest, validation-only tuning is allowed
once a set is chosen.

## 4. The 52-byte TCP floor (item 3)

The floor is applied at *extraction time*, so it changes live inputs, not training data.

| Question | Experiment |
|---|---|
| What does the floor do to benign TCP? | replay the lab-benign TCP/HTTPS sessions through the extractor with floor **on** and **off**; score with each model |
| What does it do to attacks? | replay lab-generated SYN scan, SYN flood, port sweep with floor on/off |
| Does it matter once the features change? | repeat for F0, F1, ..., F5 |

Expectation to test, not assume: for F2-F5 `MIN_IP_PKT_LEN` / `SHORTEST_FLOW_PKT` are no longer inputs, so the floor becomes
dead code and can be removed with no effect. For F0/F1 the matrix will show the false-positive vs missed-scan trade-off
quantitatively (already seen on synthetic data: 0/30 vs 30/30).

Deliverable: a floor report with FPR/recall for all combinations and a recommendation. **The floor is not changed in this
plan's analysis phase**; removal happens only together with a feature set that no longer depends on it.

## 5. Dataset composition (items 4 and 5)

### 5.1 Training sources

| Source | Content | Role |
|---|---|---|
| NF-UNSW-NB15-v3 (present, SHA recorded) | 377,279 flows; benign 249,586; attacks: Exploits 42,748, Fuzzers 33,816, Generic 19,651, Reconnaissance 17,074, DoS 5,980, Backdoor 4,659, Shellcode 2,381, Analysis 1,226, Worms 158 | attack coverage; some benign |
| **Lab benign capture (new)** | modern traffic captured with the project's own `FlowExtractor` | fills the benign gaps |
| **Lab attack capture (new, your own isolated network)** | SYN scan, port sweep, UDP flood, SYN flood, ICMP sweep, brute-force-like auth failures | measures recall on attacks that exist outside UNSW |

### 5.2 Lab benign traffic specification (proposed minimums per class)

| Class | How to generate | Proposed minimum flows |
|---|---|---|
| DNS | UDP/53 and TCP/53 queries to several resolvers (incl. 8.8.4.4, 8.8.8.8, 1.1.1.1, router) | 5,000 |
| HTTPS / TLS | scripted browsing/`curl` to varied sites (TCP/443) | 10,000 |
| QUIC / HTTP3 | browser + `curl --http3` to QUIC-enabled hosts (UDP/443) | 2,000 |
| Other TCP | HTTP, SSH, SMB, IMAP, package downloads | 5,000 |
| Other UDP | NTP (123), DHCP (67/68), mDNS (5353), SSDP (1900), NetBIOS | 2,000 |
| ICMP | ping/traceroute | 500 |
| IGMP / multicast | join/leave multicast groups, router queries | 300 (rare in practice; generated deliberately) |
| Background | normal desktop idle/updates over multiple days | remainder |

Captured over **>= 5 sessions on >= 3 different days and >= 2 devices**, so train/validation/test can be split by
session/day/device rather than by random flows.

Labelling: benign by construction (scripted, known activity on a quiescent network), then screened for contamination
(unexpected external scanning, known-bad IPs from `datasets/threat_intelligence.json`). Flows that cannot be vouched for are
excluded, not guessed. **You must confirm the capture environment and that it is your own network.**

### 5.3 ARP (and IPv6)

ARP is not an IPv4 flow, so it cannot be a row in this model and never reaches it today. The plan treats it explicitly:
* the benign capture **includes ARP**, and a test asserts it is dropped without error and counted (`pkt_errors` stays 0);
* the evaluation reports it as "out of model scope" rather than silently claiming coverage;
* a separate lightweight ARP-level check (gratuitous-ARP/duplicate-IP/MAC-flip heuristics) is listed as an **optional**
  follow-up, because ARP spoofing cannot be detected by a flow classifier. Your call whether to include it.

IGMP is IPv4 and does reach the model; it is covered by the benign capture and the per-class FPR gate.

### 5.4 Held-out evaluation sets (never used for training, tuning, or threshold choice)

| Set | Construction | Answers |
|---|---|---|
| **T1: unseen attack family** | leave-one-family-out (LOFO) over the 9 UNSW families: train on 8, test on the 9th | does it detect attack types it never saw? |
| **T2: external dataset** | a *different* NetFlow-style benchmark (see below) used **once** at the end | cross-dataset generalisation |
| **T3: lab benign, held-out sessions** | sessions/days/devices excluded from training | live false-positive rate per traffic class |
| **T4: lab attacks** | generated on your isolated network, never in training | recall on realistic attacks |
| **T5: unattended soak** | >= 2 x 1 h of normal use, no labels needed (all assumed benign) | alerts per hour in practice |

T2 (external set): investigated in [external_dataset_investigation.md](external_dataset_investigation.md). Recommended:
**NF-CSE-CIC-IDS2018-v3** (20,115,529 flows, 12.93 % attack, 2018, different organisation/testbed, 47/47 features per the
paper). Nothing has been downloaded; CSV size/header, licence fields on the record pages and the label column names are
**not verified**, and the scripted download URLs return an HTML page, not the CSV, so the file has to be fetched manually.
Evaluation is binary benign-vs-attack plus that dataset's own categories reported separately.

Split hygiene: group-based splits (by attack family, session, day, device); no flow from the same session in two splits;
thresholds and calibration fitted on a **validation** split only; a final test set is touched exactly once per candidate
pipeline and the number of looks is recorded.

## 6. Evaluation methodology (item 6)

Metrics, computed on every set above and per traffic class:

* **ROC-AUC** and **PR-AUC** (average precision), with the positive-class prevalence printed beside PR-AUC;
* at a fixed operating threshold chosen on validation: **precision, recall, F1, false-positive rate (FP/(FP+TN))**;
* **recall at fixed FPR** (0.1 %, 0.5 %, 1 %) and **FPR at fixed recall** (90 %, 95 %);
* **precision at realistic prevalence** (0.1 %, 1 %, 10 % attacks) by re-weighting, because UNSW's 34 % attack share makes
  raw precision look far better than a real network would;
* **per-class FPR table** (DNS, HTTPS, QUIC, TCP, UDP, ICMP, IGMP) with counts so tiny classes are not hidden;
* **calibration**: reliability curve, Brier score, expected calibration error (matters because thresholds are fixed numbers);
* **bootstrap 95 % confidence intervals resampled by session/family**, not by flow;
* **accuracy is reported but never used for any decision**.

Controls that guard against fooling ourselves:
* label-shuffle control (shuffled labels must give ~0.5 AUC);
* single-feature baselines (top-5 features alone) to expose any remaining shortcut;
* permutation importance on held-out data (concentration cap in section 2);
* the same evaluation harness runs the current production models as the **reference row** in every table.

## 7. Autoencoder and fusion (item 7)

Findings (script `05_ae_fusion_diagnostic.py`):

| Observation | Value |
|---|---|
| AE alone on UNSW test split | ROC-AUC 0.982, PR-AUC 0.967 (a competent anomaly detector in-distribution) |
| `ae_norm = min(mse / 51.69, 1)` | 95.7 % of flows < 0.01; benign median 7e-5. A heavy-tailed max normaliser squashes the signal into ~1e-5..1e-2 |
| XGBoost on the same split | already ROC/PR-AUC 1.0000, leaving nothing for the AE to add |
| Fusion fit | on XGBoost probabilities of the **test split**; `ae_mse_max` = 99.9th percentile of **test** MSE (leakage into calibration) |
| LR re-fits on the same data with log-scaled/standardised AE | AE weight stays ~0 or negative, because XGB is already perfect there |
| AE on live flows | median MSE 0.26-1.30 vs saved threshold 0.075: 99-100 % flagged. The AE treats *all* modern traffic as anomalous |

Conclusion: **yes, the fusion needs recalibration and retraining**, for different reasons:
1. *Wrong data*: fitted on a split where XGB is saturated, so it can never learn the AE's value (and it uses test data).
2. *Wrong encoding*: max-normalised MSE is unusable by a linear model.
3. *Wrong role*: because the AE saw only 2015 benign traffic, "high AE score" means "unlike 2015", not "attack".

Plan:
* Retrain the AE on UNSW benign **plus lab benign** (so live benign is no longer 100 % anomalous); evaluate its live FPR as
  a gate before it is allowed into fusion.
* Re-encode the AE score as a **percentile rank against the benign validation MSE** (or standardised `log1p(mse)`).
* Fit fusion on a **dedicated calibration split** (never the test sets) using **out-of-fold** XGBoost probabilities, and
  include lab-benign and lab-attack sessions so the layer sees cases where XGB is wrong.
* Compare fusion variants on the held-out sets: (a) XGB only, (b) XGB + AE (rank/log) logistic, (c) isotonic/Platt
  calibrated XGB alone, (d) **policy tiers**: critical needs XGB high; "AE high + XGB low" becomes a distinct
  *unknown/review* tier instead of a critical alert.
* Decision rule: the AE stays in the fused score only if it improves held-out PR-AUC or FPR-at-recall with a CI that
  excludes zero; otherwise it is kept only as the novelty/review signal, or removed.

## 8. Backups, separation of artifacts, reproducibility (item 8)

* **No existing file in `models/` is overwritten.** Before any training: `python scripts/preflight_retrain.py --backup`
  (copies `models/` to `models_backup/<timestamp>/`; originals untouched) and the manifest hashes are recorded.
* **Training writes only to `models_candidates/<run_id>/`**, each with its own `model_manifest.json`, `run_manifest.json`
  (git commit, dataset hashes, feature-set ID, seeds, library versions, split definitions) and a model card.
  This needs a small, reviewed change to the training scripts (`--models-dir` / `CYBERGRAPH_MODELS_DIR`, default
  unchanged). That edit is part of the plan and is **not made until you approve it**.
* The app is pointed at a candidate only via that environment variable; production `models/` stays as is. Rollback is
  unsetting the variable. Promotion (copying a candidate into `models/`) is a separate, explicitly approved step.
* Environment: `requirements-train.txt` (scikit-learn 1.8.0 etc.); one pinned set for training and serving.
  `rf_model.pkl` (legacy, 1.9.0) and `ae_model.keras` (keras 3.15.1) are not on the live path and are left alone.
* Evaluation of every candidate and the production reference runs through the same harness and the same frozen splits.

## 9. Safety, firewall, Git (items 9 and 10)

* **Firewall:** this plan performs no firewall operations. The three existing rules (`Block-GNN-175.45.176.1/.2/.3`) are not
  touched. All experiments run with the executor in its default dry-run mode; tests fake `netsh/iptables`.
  Attack generation runs only on your own isolated network and never calls the block API.
* **Git:** no commits, no pushes, no branches. Current work stays in the working tree.
* **Data handling:** lab captures stay in a gitignored folder; they may contain real hostnames/IPs, so they are not
  committed or uploaded.

## 10. Phases, gates and deliverables

| Phase | Work | Needs from you | Gate |
|---|---|---|---|
| **P0** | this plan (rev. 2) | review/approve | **G0: plan approval** (decisions received, see 12) |
| **P1** | **DONE (infrastructure):** evaluation harness + metrics + sealed test sets + splits + feature sets + synthetic traffic + candidate runner + tests. **Still to do:** feature-semantics audit write-up, freeze final splits, reference row for production models | G0 | **G1: harness review** |
| **P2** | lab capture + labelling per 5.2 (**blocked until `lab_capture_requirements.md` section 7 is signed off**); manual download + header check of T2 | sign-off; manual download of T2 | **G2: data approval** |
| **P3** | candidate models F0-F5 x 3 seeds into `models_candidates/`; LOFO; floor on/off matrix | G1, G2, approval of the training-script `--models-dir` change | **G3: results review** |
| **P4** | AE retrain, fusion recalibration variants | G3 | G4 |
| **P5** | decision report and model card; recommendation (promote / iterate / stop) | G4 | **G5: your decision on promotion** |

Deliverables
1. Evaluation harness + tests (`evaluation/`, `scripts/experiments/`), frozen split definitions, run manifests. **(built)**
2. Feature-set definitions (`config/feature_sets.py`) and the feature-semantics audit.
3. Capture tooling, labelling protocol and a data card for the lab sets.
4. Candidate models per feature set (separate directory, own manifests).
5. Comparison report: tables and plots (ROC, PR, recall-vs-FPR, per-class FPR, calibration) with confidence intervals,
   the production models as reference.
6. TCP-floor report (section 4).
7. AE/fusion report and recalibrated fusion variants (section 7).
8. Model card (intended use, data, limits) and a recommendation.
9. Updated README results section: today's README reports the near-perfect internal numbers, which should be replaced by the
   held-out numbers.
10. Backup/rollback procedure and a CI-friendly small-fixture test of the harness.

Rough effort: individual XGBoost fits take about 1-2 minutes on this machine, so the full feature-set matrix, LOFO and
fusion variants are on the order of a few hours of CPU. The long pole is **data collection and labelling**, not compute.

## 11. Risks and limits

* Lab benign comes from one household/office network; it widens coverage a lot but is still one environment. The
  soak test and the external set partly mitigate this; a second environment would help.
* Exporter-semantics mismatch between NetFlow datasets and `FlowExtractor` may limit how much of UNSW's signal transfers.
  The audit in section 3 measures this; it may force dropping more features.
* Few real attacks are available for the lab set; recall on them is a small-sample estimate and is reported with intervals.
* Some attack types may not be separable from benign using only flow features; the plan reports this rather than tuning
  until it disappears.
* IGMP and other rare classes have small counts; per-class FPR is shown with counts and wide intervals.

## 12. Your decisions (received) and what is still open

| # | Decision | Status |
|---|---|---|
| 1 | External dataset: approved, investigate first, report before downloading | **done**: [external_dataset_investigation.md](external_dataset_investigation.md); nothing downloaded |
| 2 | Lab capture: do not start; confirm requirements first; use existing data + synthetic traffic meanwhile | **honoured**: [lab_capture_requirements.md](lab_capture_requirements.md) is a sign-off checklist; no capture, no packet sent |
| 3 | Targets (provisional): FPR <= 0.5 % per flow, <= 2 false alerts/hour on the soak, >= 90 % recall on lab attacks; per-attack-family results reported separately | **adopted**; `run_candidate.py` picks thresholds at FPR 0.5 % on the calibration split and writes per-family tables |
| 4 | ARP stays outside the flow model; separate optional ARP anomaly module that cannot interfere | **built**: `modules/arp_monitor.py`, off by default, not imported by anything (a test enforces this) |
| 5 | Training scripts may write candidates to separate directories; defaults preserved; never overwrite production | **done**, with one deliberate hardening (see 13.1) |
| 6 | Storage: `latest/models_candidates/` and `latest/data/lab_captures/`, both gitignored | **done** (`config/paths.py`, `.gitignore`); I also ignored `models_backup/` (auto-backups) |

Additional requirements: separate train / validation-calibration / final-test (**built**: `evaluation/splits.py`, three disjoint
parts with recorded fingerprints); never use final test data to train or calibrate the fusion (**built**: `evaluation/sealed.py`,
enforced by `ensure_not_sealed` and covered by tests); evaluate individual models and the full fusion pipeline (**built**);
feature-set x TCP-floor combinations (**built**: `floor_matrix.py`, runs on candidates once they exist); AE on historical vs
modern benign (**done** with the production AE, see 13.3); preserve production models, keep firewall dry-run, no real firewall
actions (**honoured**; production hashes verified unchanged after every step).

Still open (need you):
* **O1. Provenance of the local UNSW file: investigated** ([external_dataset_investigation.md](external_dataset_investigation.md) section 4). It is **all** published attack flows (exact per-category match with the paper) plus **11.2 %** of the benign flows, extracted from `hybrid_nids_complete.zip` (no longer on disk), modified 2026-08-07, never tracked in Git; no published 377,279-row variant exists. Where and how it was built is **not established**; I did not read your browser history (private). Say so if you want that checked.
* **O2. External dataset: status (C), manual download needed.** `latest/data/external/` is approved and gitignored. The real CSV could not be fetched or header-checked: the link is a web app and its API answers 401 without a login. Size, header and checksum are **not verified**. Download it yourself, then run `scripts/verify_external_dataset.py` (built and tested).
* **O3. Licence: decided, academic-only.** No v3 licence text was found; the only academic-use statement is on the older v1/v2 page. Nothing assumes commercial use.
* **O4. Extractor fixes: approved as a separate reviewed change; investigation and proposal done** ([extractor_investigation_and_proposal.md](extractor_investigation_and_proposal.md)). Production `flow_extractor.py` is unchanged; the fixes are a prototype behind flags whose defaults equal today's behaviour. **Waiting for your approval of the proposal.**
* **O5. Lab capture: disabled** until `lab_capture_requirements.md` section 7 is signed off.
* **O6. Revised plan: approved. Full retraining is NOT approved.**

## 13. Revision 2: what was built and what it showed

### 13.1 Training infrastructure (all verified; production models byte-identical afterwards)

* `config/paths.py`: resolves the output directory (`--models-dir` > `CYBERGRAPH_MODELS_DIR` > production `models/`) and
  refuses to overwrite existing production models unless `--overwrite-production` (or `CYBERGRAPH_ALLOW_PRODUCTION_OVERWRITE=1`)
  is given; even then a backup goes to `models_backup/<timestamp>/` first. Candidate directories are never overwritten.
  **Deliberate hardening:** running `python train_xgboost_ae.py` with no arguments used to silently overwrite production models
  when they existed; it now refuses. Fresh setups (no models yet, `setup_pipeline.py`) behave exactly as before.
* Applied to `train_xgboost_ae.py`, `train_temporal_gnn.py`, `train_gnn.py` (CLI + function arguments) and, via the environment
  variable and the guard, to the legacy `train_fusion.py` / `train_models.py` (the legacy fusion script writes
  `fusion_model.pkl`, the same name as the live fusion model, which is exactly the accident the guard prevents).
* **Bug found and fixed on the way:** `regenerate_model_manifest()` defaulted to the *production* manifest path, so even a
  candidate run would have rewritten `models/model_manifest.json`. Manifests are now written next to the models they
  describe, and the trusted loader verifies a file against the manifest beside it (production behaviour unchanged).
* `FlowExtractor(min_pkt_floor_tcp=52)`: new constructor argument; default unchanged, `0` switches the floor off.
* `evaluation/`: `metrics.py` (precision, recall, F1, FPR, PR-AUC, ROC-AUC, recall@FPR, FPR@recall, precision at realistic
  prevalence, calibration/Brier/ECE, group-aware bootstrap CIs, per-group tables), `splits.py`, `sealed.py`,
  `synthetic_traffic.py` (in-memory only, deterministic, 12 benign + 7 attack classes through the real extractor).
* `config/feature_sets.py` (F0-F5), `scripts/experiments/run_candidate.py`, `ae_historical_vs_modern.py`, `floor_matrix.py`.
* New tests for all of the above; CI scope (bandit + flake8 gate) extended to `evaluation` and `scripts/experiments`.

Commands (smoke tests only until approval):
```
python scripts/experiments/run_candidate.py --feature-set F1 --max-rows 20000 --run-id smoke-F1
python scripts/experiments/ae_historical_vs_modern.py
python scripts/experiments/floor_matrix.py [--candidate models_candidates/<run_id> ...]
```

### 13.2 New findings about the live extractor (not model problems)

1. **Orphan flows.** The extractor exports a TCP flow at the *first* FIN/RST; the server's FIN-ACK and the client's final ACK then
   start new one-packet flows. In the earlier real capture **24 % of TCP flows (21 of 88) were such single-packet flows; the training
   data has 0.00 % (10 of 283,495)**. They are out-of-distribution live inputs.
2. **The 52-byte floor hides them.** With the floor ON these orphans score ~0.01 (0 of 21 flagged); with the true 40-byte wire length
   the production model flags **14 of 21**. So the floor masks an extractor artifact as well as the SYN-scan blind spot.
3. **Wall-clock timestamps.** `FlowExtractor` timestamps packets with `time.time()` at processing, not the packet capture time, so
   live duration/IAT/throughput include scheduling jitter whereas the training data (nProbe on pcaps) does not. A train/serve skew
   candidate for the feature-semantics audit; this is why the synthetic generator injects a controllable clock.

### 13.3 Autoencoder: historical vs modern benign (production AE, inference only)

| Domain | n | median MSE | flagged (> threshold 0.075) |
|---|---|---|---|
| HISTORICAL benign (UNSW 2015, latest 25 %) | 50,802 | 0.0035 | **1.0 %** |
| MODERN synthetic benign (all classes) **provisional** | 1,218 | 0.35 | **100 %** |
| MODERN real ad-hoc capture (presumed benign, not a lab capture) | 187 | 0.26 | **99.5 %** |

Ranking quality inside each domain: UNSW benign vs attack ROC-AUC **0.982**; synthetic modern benign vs attack ROC-AUC **0.122**
(worse than random: tiny attack flows reconstruct easily). The synthetic rows are a model of traffic; the real capture rows are
from the earlier ad-hoc capture. (Synthetic rows regenerated with the revised generator, see 13.7; the earlier 0.26 / 0.135 are superseded.) Conclusion unchanged: the AE needs modern benign data before it can play any role in fusion.

### 13.4 TCP floor x model on the same synthetic sessions (production F0, alert at 0.75; **provisional**)

| | floor ON (today) | floor OFF |
|---|---|---|
| benign false-positive rate (all) | 19.4 % | 34.2 % |
| benign TCP false-positive rate | 0.0 % | 22.0 % |
| orphan one-packet TCP flows flagged | 0.0 % | 16.7 % |
| recall: Scan / DoS / BruteForce | 21.2 % / 1.1 % / 0.0 % | 100 % / 100 % / 25.5 % |
| recall: all attacks | 13.7 % | 89.3 % |

*Regenerated with the revised generator (realistic sequence numbers and OS-dependent ACK sizes). The previous table (floor-OFF TCP FPR
67.3 %, recall 96.5 %) is **superseded**: it was inflated because every synthetic packet had sequence number 0 and every client sent bare ACKs.*

With the floor ON the production model catches almost none of the synthetic attacks; with it OFF it catches them but floods on
TCP. For feature sets without `MIN_IP_PKT_LEN`/`SHORTEST_FLOW_PKT` (F1-F5) the floor does not change the model input (verified:
identical feature matrices), i.e. it becomes dead code. The candidate models still have to be trained to fill the other rows.

### 13.5 Dataset facts

* External set: see [external_dataset_investigation.md](external_dataset_investigation.md).
* **Local training file** = **all** published attack flows (127,693, identical per category to the paper) plus **11.2 %** of the benign
  flows (249,586 of 2,237,731): attack share **33.9 % vs 5.4 % published**. (An earlier "127,639 / +54" figure was a typo on the UQ web page.)
  It came out of `hybrid_nids_complete.zip`, was modified 2026-08-07, has 200 exact duplicate rows, 3 capture days and only 35 distinct benign
  source IPs. Sampling method not determinable. Class prior in training is far from reality, so thresholds from it do not transfer.
* Benign class in the training data: 0 HTTPS, 0 QUIC, 0 NTP, 0 DHCP, 0 benign IGMP, 28 ICMP, 3 capture days (plan section 1).

### 13.6 What has NOT been done

* No full training run, no new model outside smoke tests (smoke candidates were deleted afterwards).
* No lab capture, no packet sent, no firewall action; the three existing `Block-GNN-175.45.176.*` rules are untouched.
* No download of any dataset (the external CSV needs a login; nothing was bypassed). No change to production models, thresholds, the live extractor's behaviour (default floor still 52)
  or the dashboard.
* Nothing committed or pushed.

### 13.7 Revision 3 additions

* **Extractor investigation and proposal:** [extractor_investigation_and_proposal.md](extractor_investigation_and_proposal.md). Findings: orphan flows
  after the first FIN (3.00 flows per connection; 24 % of real TCP flows single-packet vs 0.00 % in training), timing from processing time
  instead of capture time (529 us/packet measured; UDP-flood durations 93 % wrong at that cost), an unlocked shared dict (`KeyError`
  reproduced). Prototype in `proposals/extractor/` (164 added / 21 removed lines, default flags bit-identical to production), 23 tests.
  **Production `flow_extractor.py` is unchanged.** The TCP floor and the byte-rate caps are deliberately left for the model stage.
* **Synthetic generator revised** (`evaluation/synthetic_traffic.py`): realistic TCP sequence/ack numbers, OS-dependent ACK sizes (a Windows
  client sends bare 40-byte ACKs), capture timestamps on every packet, an optional processing-delay model with its own random stream.
  Earlier synthetic figures are superseded (13.3, 13.4 were regenerated).
* **External dataset verifier** (`scripts/verify_external_dataset.py`, 8 tests): rejects an HTML shell saved as `.csv`, checks the header against
  the 47 features, computes SHA-256, counts and categories. Run on the local UNSW file as a sample: 47/47 features, labels found, checksum matches.
* **Implication for the model study:** after the extractor fix with the floor OFF, Unix-style clients are 0 % flagged but Windows-style
  clients 98.9 % (the `MIN_IP_PKT_LEN` shortcut on 40-byte ACKs), so the extractor fixes and the F1-F5 study are independent and both needed.
