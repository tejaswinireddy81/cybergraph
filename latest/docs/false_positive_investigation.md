# False-positive investigation: normal traffic (e.g. 8.8.4.4) scored as critical

Status: **investigation only. No model, threshold, feature or detection rule was changed.**
All experiments are reproducible with `scripts/fp_investigation/` and write nothing under `models/`.

## Summary

The false positives are **not** caused by a wiring bug (feature names, order, scaling, dtypes all match).
They come from how the model was trained:

1. **Shortcut learning.** The label in NF-UNSW-NB15-v3 is almost a function of one header-size value,
   `MIN_IP_PKT_LEN`. XGBoost puts **79 %** of its importance on it, and a depth-2 tree on that single
   feature reaches **0.995 AUC** on the chronological test split.
2. **Out-of-distribution live traffic.** Real LAN traffic contains protocols/ports the training data never saw
   (IGMP: 33 of 377,279 training flows; UDP/443 QUIC: 0 flows) and packet sizes outside the benign range, so it lands on
   the "attack" side of the learned size thresholds.
3. **An existing patch hides part of the problem and creates a blind spot.** `flow_extractor.py` floors the TCP
   minimum packet length at 52 bytes. This makes TCP flows look benign, but it also makes a 30-port SYN scan score
   **0.002 (0 of 30 flagged)** instead of **0.998 (30 of 30)**.
4. **The held-out benchmark cannot detect any of this** (AUC 1.0000 / F1 0.9999 even after removing the top features), so
   the reported accuracy is not evidence of real-world detection quality.
5. **The autoencoder does not help the fusion layer** (logistic-regression weight on the AE score is -0.33 vs +14.76 on XGBoost).

## 1. What was ruled out (training vs live feature pipeline)

| Check | Result |
|---|---|
| `FEATURE_COLS` (47) vs `models/xgboost_ae_artifacts.json` vs `models/artifacts.json` | identical, same order |
| XGBoost booster `feature_names` vs `FEATURE_COLS` | identical (47) |
| `scaler.pkl` `feature_names_in_` vs `FEATURE_COLS` | identical (47) |
| Representation fed to XGBoost | trained on **raw** `X_train`, served **raw** (consistent) |
| Representation fed to AE/scaler | trained on scaled, served scaled (consistent) |
| dtype | float32 in training and live |
| sklearn warning "X does not have valid feature names" | harmless: scaler was fitted on a DataFrame, live passes an ndarray in the same order |

So scaling, ordering and naming are **not** the source.

## 2. The dataset artifact

NF-UNSW-NB15-v3, 377,279 flows. Attack rate by `MIN_IP_PKT_LEN` value:

| `MIN_IP_PKT_LEN` | flows | attack rate |
|---|---|---|
| 52 | 199,675 | **0.0 %** |
| 40 | 83,177 | **100 %** |
| 0 | 39,985 | 92 % |
| 73 / 65 / 76 | 26,270 / 9,540 / 6,413 | ~0 % |
| 48 | 6,152 | 100 % |

For UDP specifically: min length < 50 → 92 % attack, 50-59 → 1 %, 60-99 → 0 %, >= 100 → 97 % attack (118 flows).
TCP and UDP make up 99.4 % of the training flows (283,495 and 91,594). **IGMP (protocol 2) has only 33 flows (0.009 %)** and there are **no UDP/443 (QUIC) flows**.

This pattern is typical of traffic produced by different generator tools (bare 40-byte SYNs vs 52-byte SYNs with options),
not of attack behaviour. The model learns the tool signature.

XGBoost importances: `MIN_IP_PKT_LEN` 0.790, `DST_TO_SRC_AVG_THROUGHPUT` 0.093, `NUM_PKTS_UP_TO_128_BYTES` 0.038,
`SHORTEST_FLOW_PKT` 0.022.

## 3. Live evidence (real packets from this machine, 60 s, 213 flows, through the real `FlowExtractor`)

| Flow type | flows | flagged (XGB >= 0.75) | note |
|---|---|---|---|
| TCP | 124 | 0 | hidden by the 52-byte floor (see section 4) |
| UDP DNS (port 53) | 82 | 11 | many more sit at 0.4-0.8 |
| UDP/443 to 8.8.4.4 | 1 | 1 | prob 1.000; `MIN_IP_PKT_LEN`=50 (+6.18 log-odds) |
| IGMP / multicast | 6 | 6 | prob ~0.995-0.997; `MIN_IP_PKT_LEN` 36-40 |

* A hand-built, ideal DNS query+response through the same extractor scores **0.136** (benign). Plain DNS is not the
  main offender; short-packet UDP/QUIC and IGMP are, and DNS flows are borderline because `OUT_BYTES` and
  `NUM_PKTS_UP_TO_128_BYTES` push them up.
* The dashboard alert for `8.8.4.4` that triggered this investigation showed `MIN_IP_PKT_LEN` as the top SHAP driver
  (+5.2). That alert was not saved, so its exact flow shape could not be replayed; the reproduced case with the same
  driver is the UDP/443 flow to `8.8.4.4` above.

## 4. The existing TCP 52-byte floor (`flow_extractor.py`, present since the initial commit)

```
_MIN_PKT_FLOOR_TCP = 52
shortest_model = max(shortest, 52)   # TCP only
```

Effect measured with 30 bare 40-byte SYNs to 30 ports (a classic SYN scan, MITRE T1046, which this project claims to detect):

| | mean XGB prob | flagged >= 0.75 |
|---|---|---|
| with the floor (current behaviour) | 0.002 | **0 / 30** |
| true wire value (40) | 0.998 | 30 / 30 |

The floor suppresses false positives on TCP by rewriting the measurement, and in doing so removes the very signal the
model uses to catch scans. It also does not fix the model: with the packet-size features removed, 25-31 of the 88-124
live TCP flows are flagged instead (the model falls back on other shortcuts).

## 5. Ablation (throwaway models, same hyper-parameters and chronological 75/25 split; nothing saved)

| Variant | features | chrono-test AUC / F1 | live flagged TCP / DNS / QUIC / IGMP |
|---|---|---|---|
| A. all (reproduces production) | 47 | 1.0000 / 0.9999 | 0/124, 11/82, 1/1, 6/6 |
| B. minus `MIN_IP_PKT_LEN`, `SHORTEST_FLOW_PKT` | 45 | 1.0000 / 0.9999 | 31/124, 1/82, 0/1, 6/6 |
| C. B minus packet-size family | 38 | 1.0000 / 0.9999 | 15/124, 0/82, 0/1, 6/6 |
| D. C minus ports / DNS id+ttl / protocol ids | 29 | 1.0000 / 0.9998 | 3/124, 1/82, 0/1, 6/6 |

(A second capture reproduced the same pattern: TCP 0/88 -> 25/88 when the shortcut is removed, IGMP always flagged.)

Reading: removing one shortcut just shifts reliance to another (`DST_TO_SRC_AVG_THROUGHPUT`, `SRC_TO_DST_SECOND_BYTES`),
and the benchmark stays at ~1.0. The dataset is separable by several generator artifacts, so the chronological split does
not measure generalisation. IGMP is flagged in every variant because the model has almost never seen the protocol (33 training flows).

## 6. What is NOT recommended

* Allow-listing 8.8.4.4 or other "known good" IPs, or adding rules to hide specific alerts.
* Moving the 0.35 / 0.75 thresholds. They are applied to a score that is not calibrated for live traffic.
* Removing the TCP floor on its own. It would swap a hidden blind spot for a large number of visible TCP false positives.
* Retraining on the same data and features. That reproduces the same shortcuts.

## 7. Decisions needed before any model change

1. **Scope.** Should non-unicast / non-TCP-UDP-ICMP traffic (IGMP, multicast) be excluded from the ML score by design?
   This is a scope decision with a clear justification (almost no training data), not a hide-the-alert rule.
2. **Features.** Drop the packet-size signature family and identity-like fields (ports, DNS id/ttl) from the model input.
3. **Data.** Add labelled benign real-LAN captures (script `02_live_capture_probe.py` shows how to capture/score them) so
   the benign class covers DNS, QUIC, mDNS and IGMP/multicast traffic. Labelling must be done by you.
4. **Evaluation.** Replace the single chronological split with group-based splits (by attack family) and a cross-dataset
   test; report the live-benign false-positive rate as a first-class metric.
5. **Fusion.** Fix or remove the autoencoder branch (coefficient ~0). Re-fit fusion on data that includes live benign.
6. **TCP floor.** Remove it in the same change as the retrain, never alone.
7. **Automation.** Until live false-positive rate is measured, keep automated blocking simulated (already enforced:
   automated alerts never run real firewall commands).

## 8. Reproduce

```
python scripts/fp_investigation/01_synthetic_dns_probe.py      # ideal DNS exchange + per-feature contributions
python scripts/fp_investigation/02_live_capture_probe.py 60    # needs Npcap/admin; writes outputs/fp_live_flows.json
python scripts/fp_investigation/03_feature_ablation.py         # trains throwaway models in memory (several minutes)
python scripts/fp_investigation/04_tcp_floor_evasion_test.py   # SYN-scan with/without the 52-byte floor
```

Limitations: single machine, ~1-3 minutes of capture per run, one benign environment, and label semantics of the
dataset were inferred from value distributions (no ground truth about how each flow was generated).
