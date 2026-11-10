# CyberGraph 🛡️⚡
### Explainable AI-Powered Network Intrusion Detection & Automated Response System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green.svg)](https://xgboost.ai/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-red.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-228%20Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**CyberGraph** is a real-time, hybrid Network Intrusion Detection System (NIDS) and Security Operations Center (SOC) dashboard. It bridges modern deep learning with network security by modeling network traffic as a **Dynamic 3D Bipartite Communication Graph** and applying an ensemble of **Supervised Machine Learning, Unsupervised Anomaly Detection, and Temporal Graph Neural Networks (GNNs)**.

Unlike legacy black-box IDSs, CyberGraph provides **real-time Explainable AI (XAI)** using **SHAP (SHapley Additive exPlanations)** and topological attack-path attribution, empowering security teams to understand *why* an alert fired within milliseconds.

---

## 🚀 Key Features

* **Real-Time Live Wi-Fi Packet Sniffing**: Captures raw live 802.11/Ethernet packets via a multi-threaded Scapy engine and translates them into 47 calibrated NetFlow features on the fly.
* **Instant Auto-Detection & Fallback**: Seamlessly detects whether your device is connected to a live network. When disconnected, it automatically falls back to streaming benchmark dataset scenarios (**NF-UNSW-NB15**) with zero page refresh required.
* **3-Tier AI Detection Ensemble**:
  1. **XGBoost Classifier**: Rapid supervised classification of known attack signatures (DDoS, Exploits, Reconnaissance, Fuzzers, Backdoors).
  2. **PyTorch Deep Autoencoder**: Unsupervised anomaly detection trained strictly on benign traffic to catch zero-day attacks.
  3. **Temporal Bipartite Graph Neural Network (GNN/GAT)**: Analyzes host-to-service communication topology to flag coordinated multi-host sweeps and lateral movement.
* **Explainable AI (SHAP & Saliency)**:
  - Decomposes each critical alert into mathematical feature attributions (e.g. `MIN_IP_PKT_LEN: +6.7233`, `SHORT_FST_FLOW_PKT: +3.4879`).
  - Renders visual attribution bars and traces the complete multi-hop attack chain (`Host ➔ Service ➔ Host`).
* **Zero-Trust Device Authorization (ZTNA)**:
  - Unrecognized LAN devices connecting to the dashboard are flagged as unapproved and held in a **Pending Access Requests** queue.
  - Role-based Admin approval/revocation with cryptographic token authentication.
* **Automated Firewall Response (Dry-Run & Real Execution)**:
  - Generates exact OS firewall block rules (Windows Defender `netsh advfirewall` / Linux `iptables`).
  - Safe **Dry-Run mode** by default, with human-in-the-loop admin approval required for real firewall modifications.
* **Interactive 3D Graph Radar**:
  - Live animated bipartite radar visualizing source hosts (clients) on the left and destination services (ports) on the right, with edges color-coded by threat level.

---

## 🏗️ Architecture & Pipeline

```text
  [ Raw Network Traffic ]
      │
      ├─► Live Wi-Fi Adapter (Scapy WinPcap/Raw Socket Engine)
      └─► Offline Benchmark Stream (NF-UNSW-NB15 v3 Dataset)
            │
            ▼
  [ 1. NetFlow Feature Extractor ]
      Calculates 47 calibrated NetFlow features (ports, duration, byte/packet volume, TCP flags, TTL)
            │
            ▼
  [ 2. Bipartite Graph Construction ]
      Source Hosts (Left Nodes) ──[Flow Edges]──► Destination Services (Right Nodes)
            │
            ▼
  [ 3. Multi-Signal Detection Ensemble ]
      ├─► XGBoost (Supervised Pattern Matching)
      ├─► PyTorch Autoencoder (Unsupervised Zero-Day Anomaly Detection)
      └─► Temporal GNN / GAT (Topological Attack Coordination)
            │
            ▼
  [ 4. Fusion & Threat Scorer ]
      Weighted ensemble produces Risk Score [0.0000 – 1.0000]
      ├─ Level 1: Normal (0.00 - 0.35) ➔ Emerald Green
      ├─ Level 2: Suspicious (0.35 - 0.70) ➔ Amber Yellow
      └─ Level 3: Critical (0.70 - 1.00) ➔ Crimson Red
            │
            ▼
  [ 5. Explainable AI & Automated Response ]
      ├─► SHAP Feature Driver Breakdown & Attack Chain Path
      └─► Automated Firewall Mitigation (Dry-Run / Admin-Executed OS Rule)
```

---

## 📊 Threat Classification Levels

| Threat Level | Risk Score | UI Color | Description & Examples | Action Taken |
| :--- | :---: | :---: | :--- | :--- |
| **Level 1 (Normal)** | `0.0000` – `0.3499` | **Emerald Green** | Standard HTTP/HTTPS, DNS queries, streaming media. | Allowed; logged to rolling baseline. |
| **Level 2 (Suspicious)** | `0.3500` – `0.6999` | **Amber Yellow** | Bursty traffic, unfamiliar LAN device, minor port anomalies. | Flagged; monitored in Threat Feed. |
| **Level 3 (Critical)** | `0.7000` – `1.0000` | **Crimson Red** | SYN Floods, Port Sweeps, Exploits, DoS, Brute-Force. | Alerted with SHAP XAI card; Firewall rule generated. |

---

## 🛠️ Installation & Quickstart

### Prerequisites
- **Python**: 3.10, 3.11, or 3.12+ (tested up to Python 3.14)
- **OS**: Windows 10/11 or Linux / macOS
- **Packet Capture Driver** (for Live Wi-Fi mode): [Npcap](https://npcap.com/) (Windows) or `libpcap` (Linux)

### 1. Clone the Repository
```bash
git clone https://github.com/tejaswinireddy81/cybergraph.git
cd cybergraph
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
cd latest
pip install -r requirements.txt
```

### 4. Run CyberGraph
```bash
python app.py
```

Open your browser to:
```text
http://127.0.0.1:5000
```

---

## 📱 Multi-Device LAN Testing (Phone / Second Laptop)

To test zero-trust device detection using your phone or a secondary laptop:
1. Ensure both devices are connected to the same Wi-Fi network.
2. Check your host laptop's local IP address (e.g. `192.168.1.X` or `192.168.50.X`).
3. On your phone's browser, navigate to:
   ```text
   http://<YOUR-LAPTOP-IP>:5000
   ```
4. CyberGraph's server immediately detects the foreign IP, lists it under **"PENDING ACCESS REQUESTS"** in the bottom-left panel, and highlights it as an unapproved node on the graph radar until reviewed by an administrator.

---

## 🧪 Testing & Verification

The test suite includes unit tests, integration tests, security boundary validation, and firewall safety checks:

```bash
cd latest
python -m pytest tests/ -k "not test_browser_smoke and not test_dashboard_js" -v
```

> **Test Suite Result**: **228 passed** (100% passing across all detection, auth, and firewall suites).

---

## 📂 Project Structure

```text
cybergraph/
├── CYBERGRAPH_LIVE_DEMO_GUIDE.md     # Step-by-step demonstration manual
├── README.md                          # Project documentation
└── latest/
    ├── app.py                         # Flask server, real-time pipeline, SSE streamer
    ├── live_capture.py                # Scapy raw packet sniffer with auto-recovery
    ├── flow_extractor.py              # 47-feature NetFlow aggregation engine
    ├── requirements.txt               # Production Python dependencies
    ├── config/                        # Path and feature configuration
    ├── modules/
    │   ├── arp_monitor.py             # LAN ARP device discovery
    │   ├── auth_manager.py            # Zero-trust device auth & token verification
    │   ├── bari_engine.py             # Risk evaluation & graph construction
    │   ├── firewall_response_executor.py # Safe OS firewall executor (Dry-run enforced)
    │   └── trusted_model_loader.py    # Cryptographically verified model loading
    ├── routes/                        # Modular Flask Blueprints (Admin, Stream, Devices, Reports)
    ├── static/                        # CSS stylesheets, 3D Graph Canvas, app.js
    ├── templates/
    │   └── index.html                 # CyberGraph Command Center UI
    ├── models/                        # Pre-trained production models & scalers
    └── tests/                         # Full automated regression & safety test suite
```

---

## 📜 Dataset Reference

Trained and evaluated on **NF-UNSW-NB15-v3**, a standardized NetFlow representation of the UNSW-NB15 benchmark dataset:
- **Total Flows**: 2,365,424
- **Calibrated NetFlow Features**: 47
- **Attack Families Covered**: Exploits, Fuzzers, Generic, Reconnaissance, DoS, Backdoors, Shellcode, Analysis, Worms.
- **Evaluation Split**: Chronological temporal split (Zero future data leakage).

---

## 🛡️ License

This project is licensed under the **MIT License**.
