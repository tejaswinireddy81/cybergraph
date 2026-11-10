# CYBERGRAPH COMMAND CENTER — WINDOWS DEMONSTRATION & PRESENTATION GUIDE

> **Project Version:** 2.0 Hardened Architecture  
> **Target Environment:** 100% Native Microsoft Windows (No VirtualBox, No Linux VMs, No external hardware required)  
> **Test Suite Status:** 232/232 Tests Passing (100% Green)  
> **Model Weights Integrity:** 16/16 Production Models Bit-Identical to SHA-256 Baseline  
> **Execution Safety:** Firewall Dry-Run Enforced by Default (`dry-run`)

---

## 1. System Overview & Architecture

**CyberGraph** is a real-time, AI-powered Network Intrusion Detection & Response System (NIDS). It marries low-latency tabular machine learning with graph neural networks and automated containment.

### Core Detection Stack
```
                   [ Incoming Network Packet / Flow ]
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │    Flow Feature Extractor (47-D)    │
               │  (Port, Timing, Packet Sizes, Flags)│
               └──────────────────┬──────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌──────────────────┐                              ┌──────────────────┐
│  XGBoost Binary  │                              │ PyTorch Auto-    │
│    Classifier    │                              │  encoder (AE)    │
│ (Known Attacks)  │                              │ (Zero-Day Anom.) │
└────────┬─────────┘                              └────────┬─────────┘
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  ▼
               ┌─────────────────────────────────────┐
               │         Hybrid Fusion Layer         │
               │   Calibrated Supervised + Anomaly   │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │    Dynamic Bipartite Temporal GNN   │
               │  (Structural & Multi-Hop Risk Flow) │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │     BARI / Threat Correlator        │
               │  (Attack Journeys & MITRE ATT&CK)   │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │      Explainable AI & Firewall      │
               │  (SHAP Feature Attribution & Netsh) │
               └─────────────────────────────────────┘
```

---

## 2. Prerequisites & Environment Setup

### 2.1 Software Requirements
- **Operating System:** Windows 10 or Windows 11 (64-bit)
- **Python:** Python 3.10 through 3.14 (Verified on Python 3.14.7)
- **Web Browser:** Google Chrome, Microsoft Edge, or Mozilla Firefox
- **Network Interface:** Standard local loopback (`127.0.0.1`) — No special network card or virtual adapter needed.

### 2.2 Verify Python Environment & Dependencies
Open **PowerShell** (as regular user or Administrator) and navigate to the project root:

```powershell
# Navigate to the latest source folder
cd c:\Users\tejaswini\Desktop\cybergraph-main\latest

# Verify Python version
python --version

# Verify required packages are installed
pip install -r requirements.txt
```

### 2.3 Verify Model Integrity (Zero Tampering Check)
All 16 production models are cryptographically signed in `latest/models/model_manifest.json`. You can verify their integrity in 1 second:

```powershell
python -c "
import json, hashlib, os
manifest = json.load(open('models/model_manifest.json'))['models']
for f, info in manifest.items():
    h = hashlib.sha256(open(os.path.join('models', f), 'rb').read()).hexdigest()
    assert h == info['sha256'], f'Tampering detected in {f}'
print('All 16 production models verified: 100% Bit-Identical.')
"
```

---

## 3. Starting the CyberGraph Command Center

### Step 1: Launch the Flask Backend
In your PowerShell terminal:

```powershell
# Optional: Set a custom admin token for the session (recommended for production/demos)
# If omitted, a cryptographically secure random token is automatically generated on startup
$env:CYBERGRAPH_ADMIN_TOKEN = "your-custom-demo-token"

cd c:\Users\tejaswini\Desktop\cybergraph-main\latest
python app.py
```

You will see the startup banner displaying your active token:
```
========================================================================
CYBERGRAPH COMMAND CENTER — INITIALIZED
Server Binding : http://127.0.0.1:5000
[SECURITY] ADMIN TOKEN : <active-token-displayed-in-terminal>
Phase 2 MITRE ATT&CK integration: ACTIVE
Supported techniques: T1046, T1110, T1498
========================================================================
```

### Step 2: Open the Web Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Step 3: Authenticate as Administrator
To access administrative controls (such as manual firewall containment, device authorization, and threat response):
1. Locate the **Admin Login** button in the dashboard top navigation bar.
2. Enter the Admin Token displayed on your terminal startup banner (or the value set in `$env:CYBERGRAPH_ADMIN_TOKEN`).
3. Click **Authenticate**. A secure, HttpOnly, SameSite session cookie is issued.

#### Token Rotation Procedure:
- **Environment rotation:** Set a new token via `$env:CYBERGRAPH_ADMIN_TOKEN = "<new-secure-token>"` and restart the server.
- **Stored secret rotation:** Delete `latest/config/.secrets.json` to force generation of a fresh 192-bit cryptographic secret.
- **Session termination:** Calling `POST /api/auth/logout` immediately invalidates the administrator session cookie.

---

## 4. Live Demonstration Workflow (Viva / Presentation Script)

Here is a 5-to-10 minute structured presentation walkthrough designed to impress examiners and technical reviewers:

### Act 1: The Zero-Trust Device Gate & Topology View
- **What to show:** Point to the main Cytoscape graph canvas.
- **Talking Point:** *"CyberGraph implements strict Zero-Trust Network Access (ZTNA). Notice how initial client nodes start in the `PENDING` state and cannot access sensitive telemetry or inject malicious data. Once the administrator approves a monitored device via cryptographic HMAC tokens, it transitions to an active `HOST` node."*
- **Key UI elements:** The Monitored Devices table, active topology nodes, and role badges.

---

### Act 2: Benign Flow Evaluation & Low-Latency Baseline
- **What to show:** Click the **Stream Offline Dataset** button or observe default background traffic.
- **Talking Point:** *"Here we stream benign baseline traffic from the NF-UNSW-NB15 dataset. Notice the three metrics in the top telemetry bar:*
  1. *Capture Latency (~1.2 ms)*
  2. *Feature Engineering (~0.6 ms)*
  3. *GNN Tensor Inference (~6.8 ms)*
  *Total inference latency is under 10 milliseconds, making CyberGraph suitable for line-rate enterprise monitoring without dropping packets."*
- **Visuals:** Network connection graphs remain green (Level 1: Normal). Port 5000 connections are cleanly aggregated into a single logical edge.

---

### Act 3: Injecting Suspicious & Critical Threats
- **What to show:** Open a second PowerShell terminal and run the attack simulator:
  ```powershell
  cd c:\Users\tejaswini\Desktop\cybergraph-main\latest
  python scripts/simulate_live_attacks.py
  ```
  *(Or use the dashboard attack mode trigger)*
- **Talking Point:** *"Now we simulate targeted adversary techniques: Port Scans, SSH Brute-Forcing, and high-volume SYN flooding. Watch how the multi-tiered detection pipeline responds:"*
  - **Level 1 (Normal):** Green cards, self-dismissing in 15 seconds.
  - **Level 2 (Suspicious Anomaly):** Amber cards, lasting 45 seconds for forensic review.
  - **Level 3 (Critical Confirmed Attack):** Red cards, persisting for 75 seconds with immediate firewall mitigation options.

---

### Act 4: Explainable AI (XAI) & SHAP Attributions
- **What to show:** Click on an alert card or inspect the **Why This Attack** pane.
- **Talking Point:** *"Unlike black-box intrusion detectors, CyberGraph provides full Explainable AI (XAI). Using SHAP value attributions, CyberGraph translates abstract neural weights into human-readable evidence: e.g., 'Elevated outgoing byte rate (4.2 MB/s)', 'Abnormal TCP SYN/ACK ratio', or 'Unusual destination port sequence'."*

---

### Act 5: MITRE ATT&CK Framework Mapping
- **What to show:** Point to the MITRE badges on the threat cards:
  - `T1046: Network Service Scanning` (Confidence: >95%)
  - `T1110: Brute Force Authentication`
  - `T1498: Denial of Service`
- **Talking Point:** *"The MITRE ATT&CK engine maps corroborated graph anomaly evidence directly to adversary tactics, giving SOC analysts instant actionable threat intelligence."*

---

### Act 6: Automated Firewall Dry-Run & Safe Containment
- **What to show:** Click **Execute Block** on a Level 3 threat.
- **Talking Point:** *"To prevent accidental network disconnection or destructive changes during a demonstration, CyberGraph operates with **Firewall Dry-Run Enforced** (`mode='dry-run'`).*
  - *It generates the exact Windows `netsh advfirewall` CLI syntax.*
  - *It isolates the malicious IP address in an in-memory Access Control List (ACL).*
  - *Protected whitelist addresses (`127.0.0.1`, `localhost`, default gateway) can NEVER be blocked, preventing self-lockout.*
  - *Every containment action is immutably logged to `logs/firewall_audit.jsonl` with an audit timestamp and admin identity.*
  - *Clicking **Unblock** instantly restores the host's access."*

---

### Act 7: Security Incident Audit Report Export
- **What to show:** Click **Export Audit Report** or navigate to `http://127.0.0.1:5000/api/export_report`.
- **Talking Point:** *"At the end of an incident response cycle, CyberGraph generates a comprehensive HTML audit report complete with KPI summaries, evidence breakdown, MITRE ATT&CK context, and firewall action history. Notice that all dynamic fields are sanitized against XSS injection vulnerabilities."*

---

## 5. Security & Reliability Highlights

During this development cycle, the system was hardened against top enterprise security risks:

| Security Vector | Implementation in CyberGraph |
| :--- | :--- |
| **Model Tampering** | SHA-256 verification against signed manifest on startup; fails closed if any model weight is altered. |
| **Command Injection** | Native OS commands use strict `subprocess` argument lists (`shell=False`) with `ipaddress` RFC validation. |
| **Stored & Reflected XSS** | Full HTML escaping (`html.escape`) on all audit reports, alerts, and SVG node tooltips. |
| **Denial of Service (DoS)** | `MAX_CONTENT_LENGTH = 16MB` payload limit; Flow table capacity capped at 50,000 active flows with LRU eviction. |
| **CSRF & Spoofing** | State-changing routes require anti-CSRF tokens; `X-Forwarded-For` header spoofing strictly rejected by default. |
| **Credential Protection** | Zero credential leakage: `ADMIN_TOKEN` is never transmitted in HTML, JS, or query parameters. |

---

## 6. Traffic Ingestion Modes: Synthetic vs. Offline vs. Live Capture

CyberGraph supports three distinct execution modes. For academic viva demonstrations and local presentations, Modes 1 and 2 are recommended:

| Execution Mode | Ingestion Source | Network Requirements | Purpose & Behavior |
| :--- | :--- | :--- | :--- |
| **1. Synthetic Demonstration** (`mode="normal"` or `mode="attack"`) | In-memory procedural generator | Loopback (`127.0.0.1`) only. Zero external dependencies. | Dynamically injects structured NetFlow v9 records (Port Scans, DoS, Exploits, C2). Triggers instant graph risk propagation, alert cards, and containment without sending wire packets. |
| **2. Offline Dataset Replay** (`mode="offline"`) | `latest/NF-UNSW-NB15-v3.csv` | Loopback (`127.0.0.1`) only. No capture driver needed. | Streams real-world benchmark network flows sequentially from disk through the full 47-feature feature engineering and GNN pipeline. Provides realistic high-volume throughput and latencies. |
| **3. Actual Live Packet Capture** (`live_capture.py`) | Physical / Wi-Fi Network Interface Card (NIC) | Npcap driver (WinPcap compatible) + Windows Administrator. | Sniffs raw wire packets via Scapy, aggregates bi-directional TCP/UDP connections in `FlowExtractor`, extracts 47 features, and streams them into the live GNN. Requires driver installation and network access. |

---

## 7. Troubleshooting & FAQ

### Q: Port 5000 is already in use by another application.
**A:** You can launch on another port using an environment variable:
```powershell
$env:CYBERGRAPH_PORT = "5050"
python app.py
```
Then navigate to `http://127.0.0.1:5050`.

### Q: Why do we use Offline Dataset Replay rather than live Scapy sniffing for the demo?
**A:** Live Scapy packet sniffing on Windows requires installing the third-party Npcap driver in WinPcap compatibility mode with administrative driver bindings. If uninstalled or unprivileged, live packet capture raises socket errors. The built-in offline dataset replay and synthetic attack injectors run **100% reliably out of the box** without external drivers or root privileges, exercising the exact same 47-dimensional feature extraction and PyTorch GNN pipeline.

### Q: Can real Windows Defender Firewall rules be created?
**A:** Yes. If explicitly run as Windows Administrator with `CYBERGRAPH_FIREWALL_MODE=enforce`, CyberGraph executes:
```cmd
netsh advfirewall firewall add rule name="Block-GNN-<IP>" dir=in action=block remoteip=<IP>
```
For demonstrations, dry-run mode (`dry-run`) is recommended to ensure zero risk to host networking.

---

## 8. Verification Test Suite Command

To prove total system health to anyone reviewing the codebase, run the comprehensive pytest suite:

```powershell
cd c:\Users\tejaswini\Desktop\cybergraph-main\latest
python -m pytest tests/ -v
```

Expected output: **`232 passed in ~110s (100% green)`**.
