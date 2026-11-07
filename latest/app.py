"""
app.py
======
CYBERGRAPH COMMAND CENTER — Explainable AI Network Threat Detection & Response
Complete Implementation with Server-Authoritative Pipeline & Real-Time Synchronization
"""

import os
import json
import time
import random
import math
import logging
import copy
import threading
import uuid
from typing import Dict, Optional, List, Tuple
import numpy as np
from flask import Flask, render_template_string, jsonify, request, Response, g
from config.risk_thresholds import (
    RISK_L1_MAX,
    RISK_L2_MAX,
    XGB_PROB_HIGH,
    AE_SCORE_HIGH,
)
import socket
import ipaddress
import torch
import joblib

from modules.live_traffic_engine import LiveTrafficEngine
from modules.online_graph_stream import OnlineGraphStream
from modules.realtime_alert_engine import RealtimeAlertEngine
from modules.dynamic_temporal_gnn import DynamicBipartiteTemporalGNN
from modules.feature_schema_validator import FeatureSchemaValidator
from modules.firewall_response_executor import FirewallResponseExecutor
from modules.correlation_engine import CorrelationEngine
from modules.bari_engine import BARIEngine
from modules.attack_journey_engine import AttackJourneyEngine
from modules.explainable_graph_ai import ExplainableGraphAI
from modules.mitre_attack_mapper import MITREAttackMapper
from train_xgboost_ae import load_dataset
from live_capture import (
    start_capture,
    live_flows,
    capture_status,
    get_live_flows_snapshot,
    get_capture_status,
)

def debug_edge_decision(source_host, target_service,
                        xgb_prob, ae_score, fusion_score,
                        gnn_edge_risk, final_risk, final_tier):
    try:
        print(
            "\n"
            "================ CYBERGRAPH EDGE DIAGNOSTIC ================\n"
            f"Source       : {source_host}\n"
            f"Target       : {target_service}\n"
            f"XGBoost      : {float(xgb_prob):.6f}\n"
            f"Autoencoder  : {float(ae_score):.6f}\n"
            f"Fusion       : {float(fusion_score):.6f}\n"
            f"GNN Edge     : {float(gnn_edge_risk):.6f}\n"
            f"Final Risk   : {float(final_risk):.6f}\n"
            f"Final Tier   : {final_tier}\n"
            "-------------------------------------------------------------\n"
            f"BASE ML      : {'BENIGN' if float(fusion_score) < 0.35 else 'ELEVATED'}\n"
            f"GNN          : {'HIGH' if float(gnn_edge_risk) >= 0.75 else 'NORMAL'}\n"
            "=============================================================\n"
        )
    except Exception as e:
        print(f"[EDGE DIAG ERROR] {e}")

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
GNN_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_temporal_gnn.pt")
SCALERS_PATH = os.path.join(MODELS_DIR, "gnn_scalers.pkl")

import secrets
import hmac
import hashlib

ADMIN_TOKEN = os.environ.get("CYBERGRAPH_ADMIN_TOKEN")
if not ADMIN_TOKEN:
    ADMIN_TOKEN = secrets.token_hex(24)

HMAC_SECRET = os.environ.get("CYBERGRAPH_HMAC_SECRET")
if not HMAC_SECRET:
    HMAC_SECRET = secrets.token_hex(32)

def get_admin_token() -> str:
    """Return the active server admin authorization token."""
    global ADMIN_TOKEN
    return ADMIN_TOKEN

def generate_signed_device_token(client_ip: str) -> str:
    """Generate an HMAC-SHA256 authenticated device token bound to the client's IP."""
    dev_uuid = str(uuid.uuid4())
    ts = str(int(time.time()))
    payload = f"{client_ip}:{dev_uuid}:{ts}"
    sig = hmac.new(HMAC_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{dev_uuid}.{ts}.{sig}"

def verify_signed_device_token(token: str, client_ip: str) -> bool:
    """Cryptographically verify that the device token was signed by this server for this IP."""
    if not token or not isinstance(token, str) or "." not in token:
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    dev_uuid, ts, sig = parts
    payload = f"{client_ip}:{dev_uuid}:{ts}"
    expected_sig = hmac.new(HMAC_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return hmac.compare_digest(sig, expected_sig)

def sanitize_target_ip(raw: str) -> Optional[str]:
    """Robustly sanitize and parse an IPv4/IPv6 target address from input."""
    if not raw:
        return None
    val = str(raw).strip()
    for pfx in ("HOST:", "SERVICE:", "EXTERNAL:", "GATEWAY:", "IP:"):
        if val.upper().startswith(pfx):
            val = val[len(pfx):].strip()
    # Strip CIDR prefix length if present e.g. /32 or /128
    if "/" in val:
        val = val.split("/")[0].strip()
    # Handle bracketed IPv6 with port e.g. [2001:db8::1]:8080
    if val.startswith("[") and "]" in val:
        val = val[1:val.index("]")]
    elif ":" in val and val.count(":") == 1:
        # IPv4 with port e.g. 192.168.1.1:80
        val = val.split(":")[0].strip()
    try:
        ip_obj = ipaddress.ip_address(val)
        return str(ip_obj)
    except ValueError:
        return None

app = Flask(__name__)

# --- Server-Authoritative Shared State & Thread Lock ---
state_lock = threading.Lock()
pipeline_thread_started = False
stream_cursor = 0

shared_state = {
    "state_version": 1,
    "server_timestamp": time.time(),
    "mode": os.environ.get("CYBERGRAPH_MODE", "normal").lower() if os.environ.get("CYBERGRAPH_MODE", "").lower() in ("normal", "attack", "offline") else "normal",
    "stats": {
        "total_flows": 0,
        "level1_normal": 0,
        "level2_suspicious": 0,
        "level3_critical": 0,
        "active_hosts": 0,
        "active_services": 0,
        "real_os_rules_applied": 0,
        "current_data_source": "INITIALIZING NETWORK MONITOR...",
        "current_risk": 0.05,
        "peak_risk": 0.05,
        "risk_history": [0.05],
        "last_latency_ms": {"capture": 1.25, "feature_prep": 0.65, "gnn_infer": 6.80, "total": 8.70},
        "live_buffer_count": 0,
    },
    "latencies": {"capture": 1.25, "feature_prep": 0.65, "gnn_infer": 6.80, "total": 8.70},
    "pipeline_metrics": {
        "xgb_prob": 0.001,
        "ae_score": 0.012,
        "fusion_score": 0.001,
        "gnn_risk": 0.01,
        "final_tier": "BENIGN",
    },
    "evals": [],
    "latest_event_id": 0,
    "graph_nodes": [],
    "graph_edges": [],
    "correlation_intel": [],
    "server_status": {
        "capture_active": True,
        "processing_thread_alive": True,
        "last_packet_time": time.time(),
        "last_processing_time": time.time(),
        "state_version": 1,
        "active_clients": 1,
    },
}

traffic_engine = None
alert_engine = None
fw_executor = None
df_stream_pool = None
gnn_stream = None
correlation_engine = None
bari_engine = None
attack_journey_engine = None
mitre_mapper = None
xai_engine = None

# --- Per-flow state for pipeline explanation ---
_last_pipeline_state = {}
_last_explain_state = {}
_recent_event_history = []
_active_threat_cards = {}  # key -> {"eval": dict, "expires_at": float, "created_at": float}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Monitored Devices Registry & Access Request Queue
# ---------------------------------------------------------------------------
_authorized_monitored_devices: Dict[str, dict] = {}
_pending_device_requests: Dict[str, dict] = {}
_discovered_lan_hosts: set = set()  # set of explicitly authorized/confirmed LAN host IPs
_capture_initialized: bool = False


def init_engines():
    global traffic_engine, alert_engine, fw_executor, df_stream_pool, gnn_stream
    global correlation_engine, bari_engine, attack_journey_engine, mitre_mapper, xai_engine
    global pipeline_thread_started, _capture_initialized

    if traffic_engine is None:
        traffic_engine = LiveTrafficEngine()
        fw_executor = FirewallResponseExecutor(enable_real_blocking=False)
        alert_engine = RealtimeAlertEngine(enable_real_blocking=False, fw_executor=fw_executor)
        correlation_engine = CorrelationEngine()
        bari_engine = BARIEngine()
        attack_journey_engine = AttackJourneyEngine()
        mitre_mapper = MITREAttackMapper()
        xai_engine = ExplainableGraphAI()

        bari_baseline_path = os.path.join(MODELS_DIR, "bari_baseline.json")
        if os.path.exists(bari_baseline_path):
            bari_engine.load_baseline(bari_baseline_path)
        else:
            try:
                df_full = load_dataset()
                df_benign = df_full[df_full["label_enc"] == 0][["IPV4_SRC_ADDR", "L4_DST_PORT", "PROTOCOL"]]
                bari_engine.learn_normal_baseline(df_benign)
                bari_engine.save_baseline(bari_baseline_path)
            except Exception as e:
                logger.warning(f"Could not initialize BARI baseline from dataset: {e}")

        if df_stream_pool is None:
            df_full = load_dataset()
            df_stream_pool = df_full.iloc[int(len(df_full) * 0.75):].reset_index(drop=True)

        if os.path.exists(GNN_MODEL_PATH) and os.path.exists(SCALERS_PATH):
            scalers = joblib.load(SCALERS_PATH)
            gnn_model = DynamicBipartiteTemporalGNN(
                host_dim=8, service_dim=8, edge_dim=9, hidden_dim=32
            )
            gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH, weights_only=True))
            gnn_stream = OnlineGraphStream(gnn_model, scalers)

        try:
            start_capture(iface=None)
            _capture_initialized = True
            logger.info("Live packet capture initialized successfully.")
        except Exception as exc:
            logger.warning(f"Live capture initialization warning: {exc}")
            _capture_initialized = False

        sync_network_state(force_scan=True)
        srv_ip = get_server_local_ip()
        if srv_ip:
            fw_executor.add_protected_ip(srv_ip)
        for gw in get_default_gateways():
            fw_executor.add_protected_ip(gw)

        if not pipeline_thread_started:
            t = threading.Thread(target=_pipeline_worker, daemon=True, name="GlobalPipelineThread")
            t.start()
            pipeline_thread_started = True
            logger.info("Server-authoritative global pipeline thread launched.")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CYBERGRAPH COMMAND CENTER — Explainable AI NIDS</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-obsidian: #070a11;
            --panel-bg: rgba(15, 23, 42, 0.85);
            --panel-border: rgba(51, 65, 85, 0.6);
            --neon-cyan: #00f0ff;
            --electric-purple: #b026ff;
            --emerald-green: #00ff9d;
            --warning-amber: #ffb700;
            --critical-crimson: #ff0055;
            --mitre-blue: #4fc3f7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-obsidian);
            color: var(--text-main);
            overflow-x: hidden;
            background-image:
                radial-gradient(circle at 15% 15%, rgba(0, 240, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(176, 38, 255, 0.05) 0%, transparent 40%);
            min-height: 100vh; padding-bottom: 20px;
        }

        header {
            display: flex; justify-space-between; align-items: center;
            padding: 12px 24px; background: rgba(11, 17, 32, 0.9); backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(0, 240, 255, 0.2); position: sticky; top: 0; z-index: 1000;
        }

        .brand-container { display: flex; align-items: center; gap: 12px; }
        .brand-badge {
            width: 38px; height: 38px; background: linear-gradient(135deg, var(--neon-cyan), var(--electric-purple));
            border-radius: 8px; display: flex; align-items: center; justify-content: center;
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 20px; color: #000;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
        }

        h1 {
            font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700;
            background: linear-gradient(90deg, #ffffff, var(--neon-cyan));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }

        .sub-tag { font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

        .data-source-badge {
            font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
            padding: 6px 12px; border-radius: 6px; background: rgba(0, 240, 255, 0.1);
            border: 1px solid var(--neon-cyan); color: var(--neon-cyan); display: inline-flex; align-items: center; gap: 8px;
        }

        .dashboard-container { padding: 16px; display: flex; flex-direction: column; gap: 16px; }

        /* KPI Cards Grid */
        .kpi-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px; }
        .kpi-tile { background: var(--panel-bg); border: 1px solid var(--panel-border); border-radius: 10px; padding: 10px 12px; backdrop-filter: blur(10px); }
        .kpi-caption { font-size: 9px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
        .kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; margin-top: 4px; }

        /* Main Screen Layout */
        .main-grid { display: grid; grid-template-columns: 320px 1fr 400px; gap: 16px; align-items: start; }
        .glass-panel {
            background: var(--panel-bg); backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border); border-radius: 12px; padding: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .panel-heading {
            display: flex; justify-space-between; align-items: center;
            margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--panel-border);
        }

        .panel-title {
            font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 700;
            text-transform: uppercase; color: var(--text-muted); display: flex; align-items: center; gap: 8px;
        }

        .title-accent { width: 4px; height: 12px; background: var(--neon-cyan); border-radius: 2px; }

        .cyber-btn {
            background: linear-gradient(135deg, var(--neon-cyan), #0284c7); color: #000;
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 11px;
            text-transform: uppercase; border: none; padding: 8px 12px; border-radius: 6px;
            cursor: pointer; width: 100%; margin-top: 6px; transition: all 0.2s ease;
        }
        .cyber-btn:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0, 240, 255, 0.4); }

        .attack-btn { background: linear-gradient(135deg, var(--critical-crimson), #b91c1c); color: white; }
        .export-btn { background: linear-gradient(135deg, var(--emerald-green), #047857); color: #000; margin-top: 0; padding:6px 12px; width:auto; }
        .real-fw-btn { background: linear-gradient(135deg, var(--warning-amber), #d97706); color: #000; margin-top: 8px; font-size: 10px; }
        .unblock-btn { background: linear-gradient(135deg, var(--emerald-green), #059669); color: #000; margin-top: 6px; font-size: 10px; }

        .canvas-wrapper {
            position: relative; width: 100%; height: 750px; border-radius: 8px; overflow: hidden;
            background: radial-gradient(circle at center, #0b1120 0%, #050811 100%);
            border: 1px solid rgba(0, 240, 255, 0.15);
        }
        #graph-canvas { width: 100%; height: 100%; display: block; cursor: pointer; }

        .radar-sweep {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            border-radius: 8px; pointer-events: none;
            background: conic-gradient(from 0deg at 50% 50%, rgba(0, 240, 255, 0.08) 0deg, transparent 60deg, transparent 360deg);
            animation: sweep 6s linear infinite;
        }
        @keyframes sweep { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

        .legend-box {
            position: absolute; bottom: 10px; left: 10px; background: rgba(7, 10, 17, 0.85);
            padding: 6px 12px; border-radius: 6px; border: 1px solid var(--panel-border);
            font-family: 'JetBrains Mono', monospace; font-size: 9px; display: flex; gap: 12px;
        }
        .legend-item { display: flex; align-items: center; gap: 4px; }
        .legend-dot { width: 8px; height: 8px; border-radius: 50%; }

        .threat-feed { display: flex; flex-direction: column; gap: 10px; overflow-y: auto; height: 750px; }

        @media (max-width: 1200px) {
            .main-grid { grid-template-columns: 1fr; }
            .canvas-wrapper { height: 550px; }
            .threat-feed { height: 450px; }
        }
        .threat-card {
            background: rgba(15, 23, 42, 0.6); border-left: 4px solid var(--text-muted);
            border-radius: 6px; padding: 12px; font-size: 11px;
        }
        .threat-card.lvl-1 { border-left-color: var(--emerald-green); }
        .threat-card.lvl-2 { border-left-color: var(--warning-amber); background: rgba(255, 183, 0, 0.04); }
        .threat-card.lvl-3 { border-left-color: var(--critical-crimson); background: rgba(255, 0, 85, 0.08); }

        /* MITRE ATT&CK Card Styles */
        .mitre-section {
            margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(79, 195, 247, 0.2);
        }
        .mitre-badge {
            display: inline-block; background: rgba(79, 195, 247, 0.15);
            border: 1px solid var(--mitre-blue); border-radius: 4px;
            font-family: 'JetBrains Mono', monospace; font-size: 9px;
            color: var(--mitre-blue); padding: 2px 6px; margin: 2px;
        }
        .mitre-confidence {
            font-family: 'JetBrains Mono', monospace; font-size: 10px;
            color: var(--emerald-green); font-weight: 700;
        }
        .mitre-rationale {
            font-family: 'JetBrains Mono', monospace; font-size: 9px;
            color: var(--text-muted); margin-top: 4px; line-height: 1.4;
        }

        .why-attack-box {
            font-family: 'JetBrains Mono', monospace; background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 6px; padding: 8px;
            margin-top: 6px; font-size: 10px; color: #a5f3fc;
        }

        .why-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--neon-cyan); margin-bottom: 4px; text-transform: uppercase; }

        .cmd-box {
            font-family: 'JetBrains Mono', monospace; background: #000; color: var(--emerald-green);
            padding: 6px; border-radius: 4px; font-size: 9px; margin-top: 4px; word-break: break-all;
        }

        .sparkline-canvas { width: 100%; height: 45px; margin-top: 4px; border-radius: 6px; background: rgba(0,0,0,0.4); }

        /* Secondary Row Layout */
        .secondary-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }

        .stat-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 10px; }
        .stat-table td, .stat-table th { padding: 5px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); text-align: left; }
        .stat-table th { color: var(--text-muted); text-transform: uppercase; font-size: 9px; }

        /* System Health Grid */
        .health-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
        .health-pill { background: rgba(15, 23, 42, 0.6); border: 1px solid var(--panel-border); border-radius: 6px; padding: 6px 8px; font-family: 'JetBrains Mono', monospace; font-size: 9px; display: flex; align-items: center; justify-content: space-between; }
        .health-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--emerald-green); box-shadow: 0 0 6px var(--emerald-green); }

        /* Node Inspector Modal Drawer */
        .drawer-modal {
            display: none; position: fixed; top: 80px; right: 20px; width: 320px;
            background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(20px);
            border: 1px solid var(--neon-cyan); border-radius: 10px; padding: 14px;
            box-shadow: 0 0 25px rgba(0, 240, 255, 0.3); z-index: 2000;
            font-family: 'JetBrains Mono', monospace; font-size: 10px;
        }
        .drawer-close { float: right; cursor: pointer; color: var(--critical-crimson); font-weight: 700; }
    </style>
</head>
<body>

    <header>
        <div class="brand-container">
            <div class="brand-badge">C</div>
            <div>
                <h1>CYBERGRAPH COMMAND CENTER</h1>
                <div class="sub-tag">Explainable AI-Powered Network Threat Detection & Response</div>
            </div>
        </div>
        <div class="brand-container">
            <div class="data-source-badge" id="ds-badge">● DATA SOURCE: INITIALIZING NETWORK MONITOR...</div>
            <button class="cyber-btn" id="btn-admin-auth" style="background:rgba(124, 58, 237, 0.2); border:1px solid var(--electric-purple); color:var(--electric-purple); font-size:10px; padding:6px 10px; margin:0; width:auto;" onclick="promptAdminToken()">🔑 ADMIN AUTH</button>
            <button class="cyber-btn export-btn" onclick="window.open('/api/export_report')">Export Audit Report</button>
        </div>
    </header>

    <div class="dashboard-container">
        <!-- ROW 1: KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-tile">
                <div class="kpi-caption">Total Flows</div>
                <div class="kpi-value" id="val-total">0</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-caption">Level 1 (Normal)</div>
                <div class="kpi-value" style="color: var(--emerald-green);" id="val-l1">0</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-caption">Level 2 (Suspicious)</div>
                <div class="kpi-value" style="color: var(--warning-amber);" id="val-l2">0</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-caption">Level 3 (Critical)</div>
                <div class="kpi-value" style="color: var(--critical-crimson);" id="val-l3">0</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-caption">Active Hosts</div>
                <div class="kpi-value" style="color: var(--neon-cyan);" id="val-hosts">0</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-caption">Active Services</div>
                <div class="kpi-value" style="color: var(--electric-purple);" id="val-services">0</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-caption">Current Threat Score</div>
                <div class="kpi-value" style="color: var(--warning-amber);" id="val-risk">0.05</div>
            </div>
            <div class="kpi-tile">
                <div class="kpi-caption">Detection Latency</div>
                <div class="kpi-value" style="color: var(--emerald-green);" id="val-lat">8.70 ms</div>
            </div>
        </div>

        <!-- MAIN SCREEN LAYOUT -->
        <div class="main-grid">
            <!-- Left Panel: Threat Intelligence Controls -->
            <div class="glass-panel">
                <div class="panel-heading">
                    <span class="panel-title"><div class="title-accent"></div> Threat Intelligence Controls</span>
                    <span style="font-family:'JetBrains Mono'; font-size:9px; color:var(--text-muted);" id="version-tag">STATE V: 1</span>
                </div>

                <!-- MITRE ATT&CK Panel -->
                <div class="panel-heading" style="margin-top: 10px;">
                    <span class="panel-title"><div class="title-accent" style="background: var(--mitre-blue);"></div> MITRE ATT&CK Mapping</span>
                </div>
                <div id="mitre-panel" style="font-family:'JetBrains Mono', monospace; font-size:9px; color:var(--text-muted);">
                    <div style="padding:4px;">Awaiting detection events...</div>
                </div>

                <button class="cyber-btn" id="btn-normal" style="background: linear-gradient(135deg, var(--emerald-green), #047857); color: #000;">Live Wi-Fi (Auto-Detect)</button>
                <button class="cyber-btn" id="btn-offline" style="background: linear-gradient(135deg, #0284c7, #0369a1); color: #fff; margin-top: 6px;">Stream Offline Dataset</button>
                <button class="cyber-btn attack-btn" id="btn-attack" style="margin-top: 6px;">Replay Attack Scenario</button>

                <div class="panel-heading" style="margin-top: 10px;">
                    <span class="panel-title"><div class="title-accent" style="background: var(--warning-amber);"></div> Risk Score Trend over Time</span>
                </div>
                <canvas id="sparkline-canvas" class="sparkline-canvas"></canvas>

                <div class="panel-heading" style="margin-top: 14px;">
                    <span class="panel-title"><div class="title-accent" style="background: var(--neon-cyan);"></div> Network Interface Status</span>
                </div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:var(--text-muted); line-height:1.6; background:rgba(0,0,0,0.4); padding:10px; border-radius:6px;">
                    Physical Hardware : <span style="color:white;" id="cap-iface">Scapy WinPcap Engine</span><br>
                    Interface Status  : <span style="color:var(--emerald-green);" id="cap-status">ACTIVE</span><br>
                    Live Packet Buffer: <span style="color:var(--neon-cyan);" id="cap-buf">0 Packets</span>
                </div>

                <div class="panel-heading" style="margin-top: 14px;">
                    <span class="panel-title"><div class="title-accent" style="background: var(--emerald-green);"></div> Monitored Devices &amp; Access Requests</span>
                </div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:var(--text-muted); background:rgba(0,0,0,0.4); padding:10px; border-radius:6px; max-height:220px; overflow-y:auto;" id="device-mgmt-panel">
                    <div style="font-weight:700; color:var(--neon-cyan); margin-bottom:4px; font-size:9px;">AUTHORIZED DEVICES</div>
                    <div id="auth-device-list" style="margin-bottom:8px;">Loading...</div>
                    <div style="font-weight:700; color:var(--warning-amber); margin-bottom:4px; font-size:9px;">PENDING ACCESS REQUESTS</div>
                    <div id="pending-device-list">None</div>
                </div>
            </div>

            <!-- Center Panel: Large Live Bipartite Graph Radar -->
            <div class="glass-panel">
                <div class="panel-heading">
                    <span class="panel-title" id="graph-panel-title"><div class="title-accent"></div> Live Bipartite Graph Radar (HOST ↔ SERVICE)</span>
                </div>
                <div class="canvas-wrapper">
                    <div class="radar-sweep"></div>
                    <canvas id="graph-canvas"></canvas>
                    <div class="legend-box">
                        <div class="legend-item"><div class="legend-dot" style="background:#00f0ff;"></div> HOST (Normal)</div>
                        <div class="legend-item"><div class="legend-dot" style="background:#ffb700;"></div> HOST (Suspicious)</div>
                        <div class="legend-item"><div class="legend-dot" style="background:#ff0055;"></div> HOST (Critical)</div>
                        <div class="legend-item"><div class="legend-dot" style="background:#818cf8; border-radius:2px;"></div> GATEWAY</div>
                        <div class="legend-item"><div class="legend-dot" style="background:#b026ff;"></div> SERVICE</div>
                        <div class="legend-item"><div class="legend-dot" style="background:#ff8c00;"></div> EXTERNAL</div>
                    </div>
                </div>
            </div>

            <!-- Right Panel: Threat Feed & SHAP XAI -->
            <div class="glass-panel">
                <div class="panel-heading">
                    <span class="panel-title"><div class="title-accent" style="background: var(--critical-crimson);"></div> Threat Feed &amp; SHAP XAI</span>
                </div>
                <div class="threat-feed" id="alert-feed">
                    <!-- Injected via JS -->
                </div>
            </div>
        </div>

        <!-- ROW 3 & 4: Secondary Analytics & Firewall Controls -->
        <div class="secondary-grid">
            <!-- Temporal GNN Analytics -->
            <div class="glass-panel">
                <div class="panel-heading">
                    <span class="panel-title"><div class="title-accent" style="background: var(--electric-purple);"></div> Temporal GNN Analytics</span>
                </div>
                <table class="stat-table">
                    <tr><td>Sliding Graph Window</td><td style="color:white;">500 Flows</td></tr>
                    <tr><td>Temporal Recurrent Unit</td><td style="color:var(--neon-cyan);">GRU (h_{t-1} &rarr; h_t)</td></tr>
                    <tr><td>Node Memory State</td><td style="color:var(--emerald-green);">Active Updated</td></tr>
                    <tr><td>Temporal Convolution</td><td style="color:white;">HeteroBipartiteConv</td></tr>
                    <tr><td>Graph Edge Classifier</td><td style="color:var(--electric-purple);">EdgeRiskClassifier</td></tr>
                </table>
            </div>

            <!-- Real-Time Pipeline Latency Breakdown -->
            <div class="glass-panel">
                <div class="panel-heading">
                    <span class="panel-title"><div class="title-accent" style="background: var(--emerald-green);"></div> Real-Time Pipeline Latency</span>
                </div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:var(--text-muted); line-height:1.6;">
                    Capture Latency   : <span style="color:white;" id="lat-cap">1.25</span> ms<br>
                    Feature Prep      : <span style="color:white;" id="lat-prep">0.65</span> ms<br>
                    XGBoost / AE      : <span style="color:white;">1.80</span> ms<br>
                    Temporal GNN      : <span style="color:white;" id="lat-gnn">6.80</span> ms<br>
                    Risk & Response   : <span style="color:white;">0.20</span> ms<br>
                    <strong>Total Latency     : <span style="color:var(--emerald-green);" id="lat-tot">8.70</span> ms</strong>
                </div>
            </div>

            <!-- System Health Grid -->
            <div class="glass-panel">
                <div class="panel-heading">
                    <span class="panel-title"><div class="title-accent"></div> System Engine Health</span>
                </div>
                <div class="health-grid">
                    <div class="health-pill">Packet Cap <div class="health-dot"></div></div>
                    <div class="health-pill">Flow Extr <div class="health-dot"></div></div>
                    <div class="health-pill">XGBoost <div class="health-dot"></div></div>
                    <div class="health-pill">Autoenc <div class="health-dot"></div></div>
                    <div class="health-pill">Fusion <div class="health-dot"></div></div>
                    <div class="health-pill">TempGraph <div class="health-dot"></div></div>
                    <div class="health-pill">GNN Model <div class="health-dot"></div></div>
                    <div class="health-pill">SHAP XAI <div class="health-dot"></div></div>
                    <div class="health-pill">AttackPath <div class="health-dot"></div></div>
                    <div class="health-pill">Firewall <div class="health-dot"></div></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Node Inspector Modal Drawer -->
    <div id="inspector-modal" class="drawer-modal">
        <span class="drawer-close" onclick="closeInspector()">✕ CLOSE</span>
        <div style="font-family:'Space Grotesk'; font-size:13px; font-weight:700; color:var(--neon-cyan); margin-bottom:6px;" id="insp-title">NODE INSPECTOR</div>
        <div><strong>Entity Type:</strong> <span id="insp-type" style="color:var(--electric-purple);">HOST</span></div>
        <div style="margin-top:4px;"><strong>Entity Address:</strong> <span id="insp-addr">175.45.176.3</span></div>
        <div style="margin-top:4px;"><strong>Threat Level:</strong> <span id="insp-level" style="color:var(--emerald-green);">NORMAL</span></div>
        <div style="margin-top:4px;"><strong>Status:</strong> <span style="color:var(--emerald-green);">MONITORED & SECURED</span></div>
    </div>

    <script>
        let adminToken = localStorage.getItem('cybergraph_admin_token') || "{{ admin_token }}";
        if (adminToken) {
            localStorage.setItem('cybergraph_admin_token', adminToken);
        }

        function promptAdminToken() {
            const current = localStorage.getItem('cybergraph_admin_token') || '';
            const t = prompt('Enter CyberGraph Admin Access Token (from server startup console):', current);
            if (t !== null) {
                localStorage.setItem('cybergraph_admin_token', t.trim());
                adminToken = t.trim();
                alert('[CYBERGRAPH AUTH] Admin token saved successfully.');
                updateDeviceAccessSection();
            }
        }

        async function postAdmin(url, bodyObj = {}) {
            let token = localStorage.getItem('cybergraph_admin_token') || adminToken;
            let res;
            try {
                res = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Admin-Token': token || '',
                        'X-CSRF-Token': token || '',
                    },
                    body: JSON.stringify(bodyObj)
                });
            } catch(e) {
                throw e;
            }

            if (res.status === 403) {
                const promptToken = prompt('[SECURITY 403 FORBIDDEN]\nEnter Admin Access Token to authenticate this action:');
                if (promptToken) {
                    localStorage.setItem('cybergraph_admin_token', promptToken.trim());
                    adminToken = promptToken.trim();
                    res = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-Admin-Token': adminToken,
                            'X-CSRF-Token': adminToken,
                        },
                        body: JSON.stringify(bodyObj)
                    });
                }
            }
            return res;
        }

        const canvas = document.getElementById('graph-canvas');
        const ctx = canvas.getContext('2d');
        const sparkCanvas = document.getElementById('sparkline-canvas');
        const sctx = sparkCanvas.getContext('2d');

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
            sparkCanvas.width = sparkCanvas.parentElement.clientWidth;
            sparkCanvas.height = 45;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        let nodes = [], edges = [], particles = [];
        let nodePosMap = {};
        let riskHistory = [0.05];

        function ipToNum(ip) {
            const clean = String(ip).replace(/^(EXT|GW|PENDING):/, '');
            const parts = clean.split('.').map(Number);
            if (parts.length === 4 && !parts.some(isNaN)) {
                return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3];
            }
            return clean;
        }

        function sortNodes(nodeList) {
            return [...nodeList].sort((a, b) => {
                const numA = ipToNum(a.label);
                const numB = ipToNum(b.label);
                if (typeof numA === 'number' && typeof numB === 'number') {
                    return numA - numB;
                }
                return String(a.label).localeCompare(String(b.label));
            });
        }

        function drawSparkline() {
            sctx.clearRect(0, 0, sparkCanvas.width, sparkCanvas.height);
            if (riskHistory.length < 2) return;

            sctx.beginPath();
            const step = sparkCanvas.width / (riskHistory.length - 1);
            riskHistory.forEach((r, i) => {
                const x = i * step;
                const y = sparkCanvas.height - (r * sparkCanvas.height * 0.8 + 4);
                if (i === 0) sctx.moveTo(x, y);
                else sctx.lineTo(x, y);
            });
            sctx.strokeStyle = '#00f0ff';
            sctx.lineWidth = 2;
            sctx.stroke();
        }

        function drawStage() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const marginX = Math.max(130, Math.min(200, canvas.width * 0.28));
            const hostX = marginX;
            const serviceX = canvas.width - marginX;

            // Bipartite Column Guides
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.12)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(hostX, 0); ctx.lineTo(hostX, canvas.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(serviceX, 0); ctx.lineTo(serviceX, canvas.height); ctx.stroke();
            ctx.setLineDash([]);

            // Column Header Titles
            ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
            ctx.font = '700 9px "Space Grotesk", sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText('HOST / LAN ENDPOINTS', hostX - 16, 22);
            ctx.textAlign = 'left';
            ctx.fillText('SERVICES / TARGET PORTS', serviceX + 16, 22);

            // Draw Restrained Curved / Straight Edges
            edges.forEach(e => {
                let edgeColor = 'rgba(0, 255, 157, 0.35)';
                let lineWidth = 1.5;

                if (e.color === 'critical' || e.risk >= 0.75) {
                    edgeColor = 'rgba(255, 0, 85, 0.85)';
                    lineWidth = 2.5;
                } else if (e.color === 'suspicious' || e.risk >= 0.35) {
                    edgeColor = 'rgba(255, 183, 0, 0.65)';
                    lineWidth = 2;
                }

                // Underlay glow for critical edges
                if (e.risk >= 0.75 || e.color === 'critical') {
                    ctx.beginPath();
                    ctx.moveTo(e.x1, e.y1);
                    ctx.quadraticCurveTo(e.controlX, e.controlY, e.x2, e.y2);
                    ctx.strokeStyle = 'rgba(255, 0, 85, 0.2)';
                    ctx.lineWidth = lineWidth + 4;
                    ctx.stroke();
                }

                ctx.beginPath();
                ctx.moveTo(e.x1, e.y1);
                ctx.quadraticCurveTo(e.controlX, e.controlY, e.x2, e.y2);
                ctx.strokeStyle = edgeColor;
                ctx.lineWidth = lineWidth;
                ctx.stroke();
            });

            // Flow Particles along Bezier / Straight Curves
            particles.forEach(p => {
                p.progress += p.speed;
                if (p.progress >= 1.0) p.progress = 0.0;
                const t = p.progress;
                const invT = 1 - t;
                const currX = invT * invT * p.x1 + 2 * invT * t * p.controlX + t * t * p.x2;
                const currY = invT * invT * p.y1 + 2 * invT * t * p.controlY + t * t * p.y2;

                ctx.beginPath();
                ctx.arc(currX, currY, p.risk >= 0.75 ? 4.5 : 3, 0, Math.PI * 2);
                let pColor = '#00f0ff';
                if (p.color === 'critical' || p.risk >= 0.75) pColor = '#ff0055';
                else if (p.color === 'suspicious' || p.risk >= 0.35) pColor = '#ffb700';
                ctx.fillStyle = pColor;
                ctx.shadowColor = pColor;
                ctx.shadowBlur = 6;
                ctx.fill();
                ctx.shadowBlur = 0;
            });

            // Draw Nodes
            nodes.forEach(n => {
                const isCritical = (n.color === 'critical' || n.threat_level === 'CRITICAL');
                const isSuspicious = (n.color === 'suspicious' || n.threat_level === 'SUSPICIOUS' || n.type === 'PENDING');

                // Node radius: HOST=8, GATEWAY=8, PENDING=8, EXTERNAL=7, SERVICE=6
                const radius = (n.type === 'HOST' || n.type === 'GATEWAY' || n.type === 'PENDING') ? 8 : (n.type === 'EXTERNAL' ? 7 : 6);

                // Pulsing / glowing outer ring for Critical or Suspicious HOST / PENDING nodes
                if (n.type === 'HOST' || n.type === 'PENDING') {
                    if (isCritical) {
                        ctx.beginPath();
                        ctx.arc(n.x, n.y, radius + 5, 0, Math.PI * 2);
                        ctx.strokeStyle = 'rgba(255, 0, 85, 0.8)';
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    } else if (isSuspicious) {
                        ctx.beginPath();
                        ctx.arc(n.x, n.y, radius + 4, 0, Math.PI * 2);
                        ctx.strokeStyle = 'rgba(255, 183, 0, 0.6)';
                        ctx.lineWidth = 1.5;
                        ctx.stroke();
                    }
                }

                let nodeColor;
                if (n.type === 'SERVICE') {
                    nodeColor = '#b026ff';
                } else if (n.type === 'GATEWAY') {
                    nodeColor = '#818cf8';
                } else if (n.type === 'PENDING') {
                    nodeColor = '#ffb700'; // Amber for pending
                } else if (n.type === 'EXTERNAL') {
                    if (isCritical) nodeColor = '#ff0055';
                    else if (isSuspicious) nodeColor = '#ffb700';
                    else nodeColor = '#ff8c00';
                } else {
                    // HOST
                    if (isCritical) nodeColor = '#ff0055';
                    else if (isSuspicious) nodeColor = '#ffb700';
                    else nodeColor = '#00f0ff';
                }

                if (n.type === 'GATEWAY') {
                    // Distinct diamond/square icon for Gateway
                    ctx.beginPath();
                    ctx.rect(n.x - radius, n.y - radius, radius * 2, radius * 2);
                    ctx.fillStyle = nodeColor;
                    ctx.shadowColor = nodeColor;
                    ctx.shadowBlur = 8;
                    ctx.fill();
                    ctx.shadowBlur = 0;
                    ctx.strokeStyle = '#c7d2fe';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                } else {
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                    ctx.fillStyle = nodeColor;
                    ctx.shadowColor = nodeColor;
                    ctx.shadowBlur = (isCritical || isSuspicious || n.type === 'HOST') ? 10 : 4;
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }

                // Labels
                ctx.font = '600 11px "JetBrains Mono", monospace';
                if (n.type === 'HOST' || n.type === 'GATEWAY' || n.type === 'PENDING' || n.type === 'EXTERNAL') {
                    ctx.textAlign = 'right';
                    ctx.lineWidth = 3;
                    ctx.strokeStyle = '#070a11';

                    let displayLabel = n.label;
                    if (n.type === 'GATEWAY') {
                        displayLabel = n.label.replace(/^GW:/, '') + ' [Gateway]';
                    } else if (n.type === 'PENDING') {
                        displayLabel = n.label.replace(/^PENDING:/, '') + ' [Pending]';
                    } else if (n.type === 'EXTERNAL') {
                        displayLabel = n.label.replace(/^EXT:/, '');
                    }

                    ctx.strokeText(displayLabel, n.x - 16, n.y + 4);

                    let textColor = '#f8fafc';
                    if (n.type === 'GATEWAY') textColor = '#c7d2fe';
                    else if (n.type === 'PENDING') textColor = '#ffe066';
                    else if (isCritical) textColor = '#ff4d7d';
                    else if (isSuspicious) textColor = '#ffe066';

                    ctx.fillStyle = textColor;
                    ctx.fillText(displayLabel, n.x - 16, n.y + 4);
                } else {
                    ctx.textAlign = 'left';
                    ctx.lineWidth = 3;
                    ctx.strokeStyle = '#070a11';
                    ctx.strokeText(n.label, n.x + 16, n.y + 4);
                    ctx.fillStyle = '#e2e8f0';
                    ctx.fillText(n.label, n.x + 16, n.y + 4);
                }
            });

            requestAnimationFrame(drawStage);
        }
        requestAnimationFrame(drawStage);

        canvas.addEventListener('click', (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            nodes.forEach(n => {
                const dist = Math.hypot(n.x - mouseX, n.y - mouseY);
                if (dist <= 20) {
                    let displayAddr = n.label;
                    if (n.type === 'GATEWAY') displayAddr = n.label.replace(/^GW:/, '') + ' (Default Gateway)';
                    else if (n.type === 'PENDING') displayAddr = n.label.replace(/^PENDING:/, '') + ' (Pending Authorization)';
                    else if (n.type === 'EXTERNAL') displayAddr = n.label.replace(/^EXT:/, '');

                    document.getElementById('insp-title').innerText = `INSPECTING: ${displayAddr}`;
                    document.getElementById('insp-type').innerText = n.type;
                    document.getElementById('insp-addr').innerText = displayAddr;
                    document.getElementById('insp-level').innerText = n.threat_level || 'NORMAL';
                    document.getElementById('inspector-modal').style.display = 'block';
                }
            });
        });

        function closeInspector() {
            document.getElementById('inspector-modal').style.display = 'none';
        }

        function escapeHtml(str) {
            if (str === null || str === undefined) return '';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        function sanitizeIp(str) {
            if (!str) return '';
            return String(str).replace(/[^0-9a-zA-Z.:_-]/g, '');
        }

        async function executeRealBlock(targetIp) {
            const clean = sanitizeIp(targetIp);
            if (!confirm(`Are you sure you want to block IP ${clean} in the OS Firewall?`)) return;
            try {
                const res = await postAdmin('/api/execute_real_block', { ip: clean });
                const data = await res.json();
                if (!res.ok) {
                    alert(`[FIREWALL ACTION ERROR ${res.status}]:\n${data.message || 'Block action failed'}`);
                    return;
                }
                alert(`[ADMIN FIREWALL RESPONSE]:\n${data.message || data.status}`);
            } catch (err) {
                alert(`[FIREWALL NETWORK FAILURE]:\nUnable to reach server: ${err.message}`);
            }
        }

        async function unblockIp(targetIp) {
            const clean = sanitizeIp(targetIp);
            if (!confirm(`Are you sure you want to unblock IP ${clean}?`)) return;
            try {
                const res = await postAdmin('/api/unblock_ip', { ip: clean });
                const data = await res.json();
                if (!res.ok) {
                    alert(`[FIREWALL UNBLOCK ERROR ${res.status}]:\n${data.message || 'Unblock action failed'}`);
                    return;
                }
                alert(`[ADMIN UNBLOCK RESULT]:\n${data.message || data.status}`);
            } catch (err) {
                alert(`[UNBLOCK NETWORK FAILURE]:\nUnable to reach server: ${err.message}`);
            }
        }

        async function updateDeviceAccessSection() {
            try {
                const res = await fetch('/api/device_requests');
                if (!res.ok) {
                    console.warn(`Device requests fetch failed: HTTP ${res.status}`);
                    return;
                }
                const data = await res.json();
                const authContainer = document.getElementById('auth-device-list');
                const pendContainer = document.getElementById('pending-device-list');
                if (!authContainer || !pendContainer) return;

                const isAdmin = !!data.is_admin;

                let authHtml = '';
                if (data.authorized_devices && data.authorized_devices.length > 0) {
                    data.authorized_devices.forEach(dev => {
                        let actionHtml = '';
                        const safeDevId = escapeHtml(dev.device_id || '');
                        const safeIp = escapeHtml(dev.client_ip || '');
                        if (dev.is_server) {
                            actionHtml = '<span style="color:var(--neon-cyan); font-size:8px; font-weight:700;">SERVER</span>';
                        } else if (isAdmin) {
                            actionHtml = `<button class="cyber-btn" style="padding:2px 6px; font-size:8px; background:var(--critical-crimson); color:white; width:auto; margin:0;" onclick="revokeDevice('${safeDevId}')">REVOKE</button>`;
                        }
                        const devLabel = escapeHtml(dev.is_server ? 'CyberGraph Server' : (dev.device_name || 'Approved Device'));
                        authHtml += `
                            <div style="display:flex; justify-content:space-between; align-items:center; padding:3px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                                <div>
                                    <span style="color:var(--emerald-green); font-weight:700;">${safeIp}</span>
                                    <span style="color:var(--text-muted); font-size:8px;"> (${devLabel})</span>
                                </div>
                                <div>
                                    ${actionHtml}
                                </div>
                            </div>
                        `;
                    });
                } else {
                    authHtml = '<div style="font-size:9px; color:var(--text-muted);">No authorized devices</div>';
                }
                authContainer.innerHTML = authHtml;

                let pendHtml = '';
                if (data.pending_requests && data.pending_requests.length > 0) {
                    data.pending_requests.forEach(req => {
                        let actionHtml = '';
                        const safeReqId = escapeHtml(req.request_id || '');
                        const safeIp = escapeHtml(req.client_ip || '');
                        if (isAdmin) {
                            actionHtml = `<button class="cyber-btn" style="padding:2px 6px; font-size:8px; background:var(--emerald-green); color:black; width:auto; margin:0;" onclick="approveDevice('${safeReqId}')">APPROVE</button>`;
                        }
                        pendHtml += `
                            <div style="display:flex; justify-content:space-between; align-items:center; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                                <div>
                                    <span style="color:var(--warning-amber); font-weight:700;">${safeIp}</span>
                                    <span style="color:var(--text-muted); font-size:8px;"> PENDING</span>
                                </div>
                                <div>
                                    ${actionHtml}
                                </div>
                            </div>
                        `;
                    });
                } else {
                    pendHtml = '<div style="font-size:9px; color:var(--text-muted);">No pending requests</div>';
                }
                pendContainer.innerHTML = pendHtml;
            } catch(e) {
                console.error("Device access UI error", e);
            }
        }

        async function approveDevice(reqId) {
            try {
                const res = await postAdmin('/api/approve_device', { request_id: reqId });
                if (!res.ok) console.warn(`Approve device returned HTTP ${res.status}`);
            } catch(e) {
                console.error("Approve device error", e);
            }
            updateDeviceAccessSection();
            fetchStreamStep();
        }

        async function revokeDevice(devId) {
            try {
                const res = await postAdmin('/api/revoke_device', { device_id: devId });
                if (!res.ok) console.warn(`Revoke device returned HTTP ${res.status}`);
            } catch(e) {
                console.error("Revoke device error", e);
            }
            updateDeviceAccessSection();
            fetchStreamStep();
        }

        let isPolling = false;
        let lastDeviceStateVersion = -1;
        let lastDevicePollTime = 0;

        async function fetchStreamStep() {
            if (isPolling) return;
            isPolling = true;

            let data;
            try {
                const res = await fetch('/api/stream_step');
                if (!res.ok) {
                    console.warn(`Stream step HTTP ${res.status}`);
                    return;
                }
                data = await res.json();
            } catch(e) {
                console.error("Stream step network error:", e);
                return;
            } finally {
                isPolling = false;
            }

            if (!data || !data.stats) return;

            // Avoid unnecessary repeated device-access calls:
            // Only update device section if state_version changed or periodically after 10s
            const nowTime = Date.now();
            if (data.state_version !== lastDeviceStateVersion || (nowTime - lastDevicePollTime) > 10000) {
                lastDeviceStateVersion = data.state_version;
                lastDevicePollTime = nowTime;
                updateDeviceAccessSection();
            }

            if (data.state_version) {
                document.getElementById('version-tag').innerText = `STATE V: ${data.state_version}`;
            }

            document.getElementById('val-total').innerText = data.stats.total_flows;
            document.getElementById('val-l1').innerText = data.stats.level1_normal;
            document.getElementById('val-l2').innerText = data.stats.level2_suspicious;
            document.getElementById('val-l3').innerText = data.stats.level3_critical;
            document.getElementById('val-hosts').innerText = data.stats.active_hosts;
            document.getElementById('val-services').innerText = data.stats.active_services;
            document.getElementById('val-risk').innerText = data.stats.current_risk.toFixed(4);

            const dsBadge = document.getElementById('ds-badge');
            const graphTitle = document.getElementById('graph-panel-title');
            dsBadge.innerText = `● DATA SOURCE: ${data.stats.current_data_source}`;

            if (data.stats.live_buffer_count !== undefined) {
                document.getElementById('cap-buf').innerText = `${data.stats.live_buffer_count} Packets`;
            }

            const btnAtk = document.getElementById('btn-attack');
            const btnNorm = document.getElementById('btn-normal');
            const btnOff = document.getElementById('btn-offline');

            if (data.mode === 'attack') {
                if (btnAtk) btnAtk.style.boxShadow = '0 0 15px rgba(255, 0, 85, 0.6)';
                if (btnNorm) btnNorm.style.boxShadow = 'none';
                if (btnOff) btnOff.style.boxShadow = 'none';
            } else if (data.mode === 'offline') {
                if (btnOff) btnOff.style.boxShadow = '0 0 15px rgba(14, 165, 233, 0.8)';
                if (btnNorm) btnNorm.style.boxShadow = 'none';
                if (btnAtk) btnAtk.style.boxShadow = 'none';
            } else {
                if (btnNorm) btnNorm.style.boxShadow = '0 0 15px rgba(0, 255, 157, 0.6)';
                if (btnAtk) btnAtk.style.boxShadow = 'none';
                if (btnOff) btnOff.style.boxShadow = 'none';
            }

            function setGraphTitle(accentColor, titleText) {
                if (!graphTitle) return;
                graphTitle.textContent = ' ' + titleText;
                const acc = document.createElement('div');
                acc.className = 'title-accent';
                if (accentColor) acc.style.background = accentColor;
                graphTitle.prepend(acc);
            }

            if (data.stats.current_data_source.includes('LIVE')) {
                if (data.stats.current_data_source.includes('DISCONNECTED') || data.stats.current_data_source.includes('NO PHYSICAL')) {
                    dsBadge.style.color = 'var(--warning-amber)';
                    dsBadge.style.borderColor = 'var(--warning-amber)';
                    setGraphTitle('var(--warning-amber)', 'Live Wi-Fi Standby (No Packets Captured)');
                } else {
                    dsBadge.style.color = 'var(--emerald-green)';
                    dsBadge.style.borderColor = 'var(--emerald-green)';
                    setGraphTitle('var(--emerald-green)', 'Live Wi-Fi Bipartite Graph Radar (HOST ↔ SERVICE)');
                }
            } else if (data.stats.current_data_source.includes('ATTACK')) {
                dsBadge.style.color = 'var(--critical-crimson)';
                dsBadge.style.borderColor = 'var(--critical-crimson)';
                setGraphTitle('var(--critical-crimson)', 'Replay Attack Bipartite Graph Radar (HOST ↔ SERVICE)');
            } else {
                dsBadge.style.color = 'var(--neon-cyan)';
                dsBadge.style.borderColor = 'var(--neon-cyan)';
                setGraphTitle('', 'Offline Bipartite Graph Radar (NF-UNSW-NB15)');
            }

            if (data.latencies) {
                document.getElementById('lat-cap').innerText = data.latencies.capture.toFixed(2);
                document.getElementById('lat-prep').innerText = data.latencies.feature_prep.toFixed(2);
                document.getElementById('lat-gnn').innerText = data.latencies.gnn_infer.toFixed(2);
                document.getElementById('lat-tot').innerText = data.latencies.total.toFixed(2);
                document.getElementById('val-lat').innerText = `${data.latencies.total.toFixed(2)} ms`;
            }

            if (data.pipeline_metrics) {
                const elXgb = document.getElementById('m-xgb');
                if (elXgb) elXgb.innerText = data.pipeline_metrics.xgb_prob.toFixed(4);
                const elAe = document.getElementById('m-ae');
                if (elAe) elAe.innerText = data.pipeline_metrics.ae_score.toFixed(4);
                const elFus = document.getElementById('m-fus');
                if (elFus) elFus.innerText = data.pipeline_metrics.fusion_score.toFixed(4);
                const elGnn = document.getElementById('m-gnn');
                if (elGnn) elGnn.innerText = data.pipeline_metrics.gnn_risk.toFixed(4);
                const elFin = document.getElementById('m-fin');
                if (elFin) elFin.innerText = data.pipeline_metrics.final_tier;
            }

            if (data.stats.risk_history) {
                riskHistory = data.stats.risk_history;
                drawSparkline();
            }

            const feed = document.getElementById('alert-feed');
            feed.innerHTML = '';

            data.evals.forEach(ev => {
                const isNormal = ev.level && ev.level.includes('Level 1');
                const card = document.createElement('div');
                card.className = `threat-card ${isNormal ? 'lvl-1' : (ev.level.includes('Level 2') ? 'lvl-2' : 'lvl-3')}`;

                const safeLvl = escapeHtml(ev.level || 'Unknown');
                const safeBadge = escapeHtml(ev.badge_color || '#10b981');
                const safeSrc = escapeHtml(ev.source_host || '-');
                const safeDst = escapeHtml(ev.target_service || '-');
                const safeSummary = escapeHtml(ev.summary || '');
                const safeRisk = Number(ev.risk_score || 0).toFixed(4);

                let html = `
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <strong>${safeLvl}</strong>
                        <span style="background:${safeBadge}; color:#000; padding:2px 6px; border-radius:4px; font-weight:700;">Risk: ${safeRisk}</span>
                    </div>
                    <div><strong>Src:</strong> ${safeSrc} &rarr; <strong>Dst:</strong> ${safeDst}</div>
                    <div style="margin-top:4px; color:var(--text-muted);">${safeSummary}</div>
                `;

                if (!isNormal && ev.xai_explanation && ev.xai_explanation.why_this_attack_reasons) {
                    const reasons = ev.xai_explanation.why_this_attack_reasons;
                    html += `
                        <div class="why-attack-box">
                            <div class="why-title">WHY THIS ATTACK? (SHAP & SALIENCY)</div>
                            ${reasons.map(r => `<div>${escapeHtml(r)}</div>`).join('')}
                        </div>
                    `;
                }

                if (!isNormal && ev.bipartite_attack_chain) {
                    const safeChain = ev.bipartite_attack_chain.map(c => escapeHtml(c)).join(' &rarr; ');
                    html += `<div style="margin-top:6px; font-family:'JetBrains Mono'; font-size:10px; color:#ff99bb;"><strong>Attack Chain:</strong><br>${safeChain}</div>`;
                }

                if (!isNormal && ev.mitre_mappings && ev.mitre_mappings.length > 0) {
                    html += `<div class="mitre-section">`;
                    html += `<div style="color:var(--mitre-blue); font-weight:700; margin-bottom:4px;">MITRE ATT&CK Techniques Detected:</div>`;
                    ev.mitre_mappings.forEach(m => {
                        const safeTechId = escapeHtml(m.technique_id || '');
                        const safeTechName = escapeHtml(m.technique_name || '');
                        const safeConf = (Number(m.confidence || 0) * 100).toFixed(1);
                        const safeRationale = escapeHtml(m.rationale || '');
                        const safeTactic = escapeHtml(m.tactic || '');

                        html += `<div style="margin:4px 0;">`;
                        html += `<span class="mitre-badge">${safeTechId}</span> `;
                        html += `<span style="color:var(--text-main);">${safeTechName}</span> `;
                        html += `<span class="mitre-confidence">${safeConf}%</span>`;
                        html += `</div>`;
                        if (safeRationale) {
                            html += `<div class="mitre-rationale">${safeRationale}</div>`;
                        }
                        if (safeTactic) {
                            html += `<div style="color:var(--text-muted); font-size:8px;">Tactic: ${safeTactic}</div>`;
                        }
                    });
                    html += `</div>`;
                }

                if (!isNormal && ev.recommended_firewall_action) {
                    const fw = ev.recommended_firewall_action;
                    const ipClean = sanitizeIp(fw.target_ip || '');
                    const safePreview = escapeHtml(fw.windows_netsh_rule_preview || '');
                    html += `
                        <div style="margin-top:8px;">
                            <strong style="color:var(--critical-crimson);">Recommended Block Action (Simulated Response):</strong>
                            <div class="cmd-box">${safePreview}</div>
                            <button class="cyber-btn real-fw-btn" onclick="executeRealBlock('${ipClean}')">Admin Approval: Execute Real OS Firewall Rule</button>
                            <button class="cyber-btn unblock-btn" onclick="unblockIp('${ipClean}')">Admin Action: Unblock IP</button>
                        </div>
                    `;
                }

                card.innerHTML = html;
                feed.appendChild(card);
            });

            // PERFECT DYNAMIC BIPARTITE GRAPH FIT
            // Left column: HOST + GATEWAY + EXTERNAL nodes  |  Right column: SERVICE nodes
            const hostNodes = sortNodes(data.graph_nodes.filter(n => n.type === 'HOST'));
            const gatewayNodes = sortNodes(data.graph_nodes.filter(n => n.type === 'GATEWAY'));
            const externalNodes = sortNodes(data.graph_nodes.filter(n => n.type === 'EXTERNAL'));
            const serviceNodes = sortNodes(data.graph_nodes.filter(n => n.type === 'SERVICE'));

            // All left-side nodes combined (HOSTs first, then GATEWAY, then EXTERNALs)
            const leftNodes = [...hostNodes, ...gatewayNodes, ...externalNodes];

            const marginX = Math.max(140, Math.min(220, canvas.width * 0.28));
            const hostX = marginX;
            const serviceX = canvas.width - marginX;

            const topPad = 60;
            const bottomPad = 60;
            const availH = canvas.height - topPad - bottomPad;

            const lCount = leftNodes.length;
            const sCount = serviceNodes.length;

            const newNodePosMap = {};

            // Distribute all left-column nodes (HOST + GATEWAY + EXTERNAL) vertically with generous padding
            leftNodes.forEach((n, idx) => {
                const y = lCount <= 1 ? canvas.height / 2 : topPad + idx * (availH / (lCount - 1));
                newNodePosMap[n.label] = { x: hostX, y: y };
            });

            serviceNodes.forEach((s, idx) => {
                const y = sCount <= 1 ? canvas.height / 2 : topPad + idx * (availH / (sCount - 1));
                newNodePosMap[s.label] = { x: serviceX, y: y };
            });

            nodePosMap = newNodePosMap;

            nodes = data.graph_nodes.map(n => ({
                label: n.label,
                type: n.type,
                threat_level: n.threat_level || 'NORMAL',
                color: n.color || 'normal',
                x: nodePosMap[n.label] ? nodePosMap[n.label].x : ((n.type === 'HOST' || n.type === 'GATEWAY' || n.type === 'EXTERNAL') ? hostX : serviceX),
                y: nodePosMap[n.label] ? nodePosMap[n.label].y : canvas.height / 2,
            }));

            // Restrained Edge Routing: Prefer mostly straight lines; small ±12px offset only for multiple overlapping edges between same pair
            const edgePairCounts = {};
            const edgePairIndices = {};

            data.graph_edges.forEach(e => {
                const pairKey = `${e.source_label}->${e.target_label}`;
                edgePairCounts[pairKey] = (edgePairCounts[pairKey] || 0) + 1;
            });

            edges = data.graph_edges.map(e => {
                const pairKey = `${e.source_label}->${e.target_label}`;
                const idx = edgePairIndices[pairKey] || 0;
                edgePairIndices[pairKey] = idx + 1;
                const total = edgePairCounts[pairKey];

                const srcPos = nodePosMap[e.source_label] || { x: hostX, y: canvas.height / 2 };
                const dstPos = nodePosMap[e.target_label] || { x: serviceX, y: canvas.height / 2 };

                const midX = (srcPos.x + dstPos.x) / 2;
                const midY = (srcPos.y + dstPos.y) / 2;

                // Restrained curve offset: 0 for single edges (straight line!), ±14px max for duplicate parallel edges
                let curveOffset = 0;
                if (total > 1) {
                    const rawOffset = (idx - (total - 1) / 2) * 14;
                    curveOffset = Math.max(-18, Math.min(18, rawOffset));
                }

                return {
                    x1: srcPos.x, y1: srcPos.y,
                    x2: dstPos.x, y2: dstPos.y,
                    controlX: midX,
                    controlY: midY + curveOffset,
                    risk: e.risk, color: e.color || 'normal', level: e.level || 'NORMAL'
                };
            });

            particles = edges.map(e => ({
                x1: e.x1, y1: e.y1, x2: e.x2, y2: e.y2,
                controlX: e.controlX, controlY: e.controlY,
                risk: e.risk, color: e.color,
                progress: Math.random(), speed: 0.004 + e.risk * 0.012
            }));
        }

        document.getElementById('btn-normal').addEventListener('click', async () => {
            await postAdmin('/api/set_mode', { mode: 'normal' });
            await fetchStreamStep();
        });
        const btnOfflineEl = document.getElementById('btn-offline');
        if (btnOfflineEl) {
            btnOfflineEl.addEventListener('click', async () => {
                await postAdmin('/api/set_mode', { mode: 'offline' });
                await fetchStreamStep();
            });
        }
        document.getElementById('btn-attack').addEventListener('click', async () => {
            await postAdmin('/api/set_mode', { mode: 'attack' });
            await fetchStreamStep();
        });

        function scheduleNextPoll() {
            setTimeout(async () => {
                await fetchStreamStep();
                scheduleNextPoll();
            }, 750);
        }

        fetchStreamStep().finally(() => {
            scheduleNextPoll();
        });
    </script>
</body>
</html>
"""


def is_live_capture_available() -> bool:
    """Return True if Scapy/Npcap live capture is initialized and available.

    Live capture is considered unavailable ONLY if:
      - Initialization failed at startup (_capture_initialized is False), OR
      - Capture thread exited with a fatal error (capture_status['last_error'] is set
        and capture_status['running'] is False).
    A temporary lack of physical network packets does NOT make live capture unavailable.
    """
    if not _capture_initialized:
        return False
    if capture_status.get("last_error") and not capture_status.get("running", False):
        return False
    return bool(capture_status.get("running", False) or _capture_initialized)


def get_recent_live_flows(window_seconds: float = 10.0):
    """Returns physical network flows captured within window_seconds in a thread-safe manner."""
    return get_live_flows_snapshot(window_seconds=window_seconds)


def check_active_live_traffic(window_seconds: float = 10.0):
    """Backwards-compatible helper: returns (has_recent_traffic, recent_flows)."""
    recent = get_recent_live_flows(window_seconds)
    return (len(recent) > 0), recent



def _classify_attack_type(flow_rec):
    """Heuristic attack type classification for correlation engine."""
    xgb_prob = flow_rec.get("xgb_prob", 0.0)
    ae_score = flow_rec.get("ae_score", 0.0)
    fusion_score = flow_rec.get("fusion_score", 0.0)
    dst_port = flow_rec.get("dst_port", 0)
    pkt_count = flow_rec.get("pkt_count", 0)
    duration = flow_rec.get("duration", 1.0)

    if xgb_prob >= XGB_PROB_HIGH or ae_score >= AE_SCORE_HIGH:
        return "Exploits"
    elif pkt_count / max(duration, 0.001) > 500:
        return "DoS"
    elif dst_port in (21, 22, 23, 3389) and xgb_prob >= 0.4:
        return "Reconnaissance"
    elif dst_port in (4444, 5555, 31337):
        return "Backdoor"
    elif fusion_score >= 0.6 and ae_score >= 0.01:
        return "Fuzzer"
    else:
        return "Normal"


def _lookup_geo(ip):
    """Cached simple geolocation lookup."""
    return {"country": "Unknown", "asn": "N/A"}


# ---------------------------------------------------------------------------
# Local LAN Monitored Subnet & Host Classification Helper — VISUALIZATION ONLY
# Does NOT affect ML pipeline, detection, or risk scoring in any way.
# ---------------------------------------------------------------------------
import ipaddress as _ipaddress
from typing import Optional, Dict, List, Set, Tuple

_RFC1918_NETWORKS = (
    _ipaddress.IPv4Network("10.0.0.0/8"),
    _ipaddress.IPv4Network("172.16.0.0/12"),
    _ipaddress.IPv4Network("192.168.0.0/16"),
)

_monitored_local_networks: set = set()
_default_gateways: set = set()
_server_local_ip: Optional[str] = None
_current_network_id: Optional[tuple] = None
_last_neighbor_scan_time: float = 0.0


def discover_default_gateways() -> set:
    """Dynamically discover active IPv4 default gateway addresses.

    Uses Scapy routing table and system netstat / route print tables.
    Excludes loopback (127.x.x.x), link-local (169.254.x.x), multicast, 0.0.0.0, and broadcast.
    Zero hardcoded IP addresses.
    """
    gws = set()
    try:
        from scapy.all import conf
        def_route = conf.route.route("0.0.0.0")
        if def_route and len(def_route) >= 3:
            gw_candidate = str(def_route[2])
            if gw_candidate and gw_candidate not in ("0.0.0.0", "127.0.0.1"):
                try:
                    ip_obj = _ipaddress.IPv4Address(gw_candidate)
                    if (
                        not ip_obj.is_loopback
                        and not ip_obj.is_link_local
                        and not ip_obj.is_multicast
                        and not ip_obj.is_unspecified
                    ):
                        gws.add(gw_candidate)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        from scapy.all import conf
        for r in conf.route.routes:
            if len(r) >= 3:
                gw = str(r[2])
                if gw and gw not in ("0.0.0.0", "127.0.0.1", "255.255.255.255"):
                    try:
                        ip_obj = _ipaddress.IPv4Address(gw)
                        if (
                            not ip_obj.is_loopback
                            and not ip_obj.is_link_local
                            and not ip_obj.is_multicast
                            and not ip_obj.is_unspecified
                        ):
                            gws.add(gw)
                    except Exception:
                        pass
    except Exception:
        pass

    if not gws:
        try:
            import subprocess
            out = subprocess.check_output(
                ["netstat", "-rn"], shell=False, text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] == "0.0.0.0":
                    gw = parts[2]
                    try:
                        ip_obj = _ipaddress.IPv4Address(gw)
                        if (
                            not ip_obj.is_loopback
                            and not ip_obj.is_link_local
                            and not ip_obj.is_multicast
                            and not ip_obj.is_unspecified
                        ):
                            gws.add(gw)
                    except Exception:
                        pass
        except Exception:
            pass

    return gws


def get_default_gateways() -> set:
    """Return the set of dynamically discovered default gateway IP strings."""
    global _default_gateways, _active_network_profile_override
    if not _default_gateways:
        if _active_network_profile_override and _active_network_profile_override.get("default_gateway"):
            _default_gateways = {str(_active_network_profile_override["default_gateway"])}
        else:
            _default_gateways = discover_default_gateways()
    return _default_gateways


def set_default_gateways(gateways):
    """Explicitly set or override the default gateway IPs (e.g. for testing or reconfiguration)."""
    global _default_gateways
    _default_gateways = set(str(g) for g in gateways)


def add_default_gateway(gateway: str):
    """Add a gateway IP string to the set of default gateways."""
    global _default_gateways
    _default_gateways.add(str(gateway))


def discover_local_networks() -> set:
    """Dynamically discover IPv4 local LAN subnets from active network interfaces.

    Inspects active network interfaces using psutil and Scapy.
    Constructs IPv4Network objects using interface IPv4 and netmask.
    Excludes loopback (127.0.0.0/8), link-local (169.254.0.0/16), and non-RFC1918 addresses.
    Zero hardcoded IP addresses.
    """
    discovered = set()
    try:
        import psutil
        for iface_name, addrs in psutil.net_if_addrs().items():
            for snic in addrs:
                # AF_INET family is 2
                if getattr(snic, "family", None) == 2:
                    try:
                        ip_obj = _ipaddress.IPv4Address(snic.address)
                        if (
                            any(ip_obj in rfc_net for rfc_net in _RFC1918_NETWORKS)
                            and not ip_obj.is_loopback
                            and not ip_obj.is_link_local
                            and not ip_obj.is_multicast
                            and not ip_obj.is_unspecified
                        ):
                            mask = snic.netmask or "255.255.255.0"
                            net = _ipaddress.IPv4Network(f"{snic.address}/{mask}", strict=False)
                            discovered.add(net)
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        from scapy.all import conf
        scapy_ip = getattr(conf.iface, "ip", None)
        if scapy_ip:
            ip_obj = _ipaddress.IPv4Address(scapy_ip)
            if (
                any(ip_obj in rfc_net for rfc_net in _RFC1918_NETWORKS)
                and not ip_obj.is_loopback
                and not ip_obj.is_link_local
            ):
                netmask = getattr(conf.iface, "netmask", None) or "255.255.255.0"
                discovered.add(_ipaddress.IPv4Network(f"{scapy_ip}/{netmask}", strict=False))
    except Exception:
        pass

    return discovered


def get_monitored_local_networks() -> set:
    """Return the currently monitored local IPv4 networks, discovering if empty."""
    global _monitored_local_networks
    if not _monitored_local_networks:
        _monitored_local_networks = discover_local_networks()
    return _monitored_local_networks


def set_monitored_local_networks(networks):
    """Explicitly set or override the monitored local networks (e.g. for testing or reconfiguration).

    Accepts an iterable of IPv4Network objects or CIDR strings (e.g. ["192.168.1.0/24"]).
    """
    global _monitored_local_networks
    new_set = set()
    for item in networks:
        if isinstance(item, _ipaddress.IPv4Network):
            new_set.add(item)
        elif isinstance(item, str):
            new_set.add(_ipaddress.IPv4Network(item, strict=False))
    _monitored_local_networks = new_set


def add_monitored_local_network(network):
    """Add a network to the monitored local networks."""
    global _monitored_local_networks
    if isinstance(network, _ipaddress.IPv4Network):
        _monitored_local_networks.add(network)
    elif isinstance(network, str):
        _monitored_local_networks.add(_ipaddress.IPv4Network(network, strict=False))


def get_active_network_profile() -> dict:
    """Dynamically determine active network interface, local IPv4, subnet, and default gateway.

    Selection Priority:
    1. Environment overrides: CYBERGRAPH_SERVER_IP, CYBERGRAPH_SUBNET, CYBERGRAPH_GATEWAY, CYBERGRAPH_IFACE
    2. Physical capture interface (matched against lowest metric default route on active non-virtual interface)
    3. Fail-safe validation to prevent auto-authorizing wrong virtual/secondary IPs.
    """
    env_ip = os.environ.get("CYBERGRAPH_SERVER_IP")
    env_subnet = os.environ.get("CYBERGRAPH_SUBNET")
    env_gw = os.environ.get("CYBERGRAPH_GATEWAY")
    env_iface = os.environ.get("CYBERGRAPH_IFACE")

    if env_ip:
        sub_obj = None
        if env_subnet:
            try:
                sub_obj = _ipaddress.IPv4Network(env_subnet, strict=False)
            except Exception:
                sub_obj = _ipaddress.IPv4Network(f"{env_ip}/24", strict=False)
        else:
            sub_obj = _ipaddress.IPv4Network(f"{env_ip}/24", strict=False)

        profile = {
            "iface": env_iface or "Manual-Config",
            "local_ip": env_ip,
            "netmask": "255.255.255.0",
            "local_subnet": sub_obj,
            "default_gateway": env_gw or None,
            "capture_iface": env_iface or "Manual",
            "consistent": True,
        }
        logger.info(
            f"[ACTIVE MONITORING PROFILE - ENV OVERRIDE]\n"
            f"iface={profile['iface']}\n"
            f"capture_iface={profile['capture_iface']}\n"
            f"server_local_ip={profile['local_ip']}\n"
            f"subnet={profile['local_subnet']}\n"
            f"gateway={profile['default_gateway']}\n"
            f"consistent=True"
        )
        return profile

    local_ip = None
    default_gateway = None
    iface_name = None
    netmask = None
    capture_iface_str = "Default"

    # Virtual adapter keywords to exclude from physical capture selection
    virtual_keywords = (
        "virtual", "vethernet", "loopback", "wsl", "host-only",
        "vmware", "virtualbox", "direct virtual", "wan miniport",
        "bluetooth", "teredo", "isatap", "hyper-v", "docker", "vpn"
    )

    candidate_log = []

    # Step 1: Query Scapy conf.iface and conf.route
    scapy_cand_ip = None
    scapy_cand_gw = None
    try:
        from scapy.all import conf
        if hasattr(conf, "iface") and conf.iface:
            capture_iface_str = getattr(conf.iface, "name", str(conf.iface))
            cand_ip = getattr(conf.iface, "ip", None)
            if cand_ip and str(cand_ip) not in ("0.0.0.0", "127.0.0.1"):
                scapy_cand_ip = str(cand_ip)
        def_route = conf.route.route("0.0.0.0")
        if def_route and len(def_route) >= 3:
            if str(def_route[1]) not in ("0.0.0.0", "127.0.0.1"):
                scapy_cand_ip = str(def_route[1])
            if str(def_route[2]) not in ("0.0.0.0", "127.0.0.1", "255.255.255.255"):
                scapy_cand_gw = str(def_route[2])
    except Exception as exc:
        logger.debug(f"Scapy route detection exception: {exc}")

    # Step 2: Query system routing table (netstat -rn) to find active default route with lowest metric
    best_route_metric = 999999
    netstat_cand_ip = None
    netstat_cand_gw = None
    try:
        import subprocess
        out = subprocess.check_output(
            ["netstat", "-rn"], shell=False, text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "0.0.0.0" and parts[1] == "0.0.0.0":
                gw = parts[2]
                lip = parts[3]
                metric = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else 100
                if gw not in ("0.0.0.0", "127.0.0.1", "255.255.255.255") and lip not in ("0.0.0.0", "127.0.0.1"):
                    candidate_log.append(f"netstat default route: IP={lip}, GW={gw}, Metric={metric}")
                    if metric < best_route_metric:
                        best_route_metric = metric
                        netstat_cand_gw = gw
                        netstat_cand_ip = lip
    except Exception as exc:
        logger.debug(f"Netstat route detection exception: {exc}")

    # Step 3: Query OS UDP routing table lookup
    sock_cand_ip = None
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        sip = s.getsockname()[0]
        s.close()
        if sip not in ("0.0.0.0", "127.0.0.1"):
            sock_cand_ip = sip
            candidate_log.append(f"socket default route query: IP={sock_cand_ip}")
    except Exception:
        pass

    # Step 4: Match with psutil for physical interfaces
    psutil_candidates = []
    try:
        import psutil
        for name, addrs in psutil.net_if_addrs().items():
            is_virtual = any(kw in name.lower() for kw in virtual_keywords)
            for snic in addrs:
                if getattr(snic, "family", None) == 2:  # AF_INET
                    addr_str = snic.address
                    try:
                        ip_obj = _ipaddress.IPv4Address(addr_str)
                        if (
                            not ip_obj.is_loopback
                            and not ip_obj.is_link_local
                            and not ip_obj.is_multicast
                            and not ip_obj.is_unspecified
                            and any(ip_obj in rfc for rfc in _RFC1918_NETWORKS)
                        ):
                            psutil_candidates.append({
                                "name": name,
                                "ip": addr_str,
                                "netmask": snic.netmask or "255.255.255.0",
                                "is_virtual": is_virtual,
                            })
                    except Exception:
                        pass
    except Exception as exc:
        logger.debug(f"psutil detection exception: {exc}")

    for c in psutil_candidates:
        candidate_log.append(f"psutil candidate: name='{c['name']}', IP={c['ip']}, netmask={c['netmask']}, virtual={c['is_virtual']}")

    # Decision logic:
    target_ip = netstat_cand_ip or scapy_cand_ip or sock_cand_ip
    target_gw = netstat_cand_gw or scapy_cand_gw

    selected_cand = None
    if target_ip:
        # First try to find a NON-VIRTUAL psutil candidate matching target_ip
        for c in psutil_candidates:
            if c["ip"] == target_ip and not c["is_virtual"]:
                selected_cand = c
                break
        if not selected_cand:
            for c in psutil_candidates:
                if c["ip"] == target_ip:
                    selected_cand = c
                    break

    # If no match yet, pick first NON-VIRTUAL psutil candidate
    if not selected_cand:
        for c in psutil_candidates:
            if not c["is_virtual"]:
                selected_cand = c
                break

    # Only if absolutely no non-virtual adapter exists, pick first available
    if not selected_cand and psutil_candidates:
        selected_cand = psutil_candidates[0]

    if selected_cand:
        local_ip = selected_cand["ip"]
        iface_name = selected_cand["name"]
        netmask = selected_cand["netmask"]
        default_gateway = target_gw
    else:
        local_ip = target_ip
        default_gateway = target_gw
        iface_name = "Wi-Fi"
        netmask = "255.255.255.0"

    local_subnet = None
    if local_ip:
        mask_to_use = netmask or "255.255.255.0"
        try:
            local_subnet = _ipaddress.IPv4Network(f"{local_ip}/{mask_to_use}", strict=False)
        except Exception:
            local_subnet = _ipaddress.IPv4Network(f"{local_ip}/24", strict=False)

    consistent = True
    if scapy_cand_ip and local_ip:
        consistent = (scapy_cand_ip == local_ip)

    profile = {
        "iface": iface_name or "Wi-Fi",
        "local_ip": local_ip,
        "netmask": netmask or "255.255.255.0",
        "local_subnet": local_subnet,
        "default_gateway": default_gateway,
        "capture_iface": capture_iface_str,
        "consistent": consistent,
    }

    logger.info(
        f"[ACTIVE MONITORING PROFILE]\n"
        f"iface={profile['iface']}\n"
        f"capture_iface={profile['capture_iface']}\n"
        f"server_local_ip={profile['local_ip']}\n"
        f"subnet={profile['local_subnet']}\n"
        f"gateway={profile['default_gateway']}\n"
        f"consistent={profile['consistent']}"
    )
    if candidate_log:
        logger.info("[INTERFACE CANDIDATES EVALUATED]\n" + "\n".join(candidate_log))

    return profile


def discover_lan_neighbors(profile: Optional[dict] = None, arp_output: Optional[str] = None) -> set:
    """Dynamically discover confirmed LAN neighbor/endpoint IPv4 addresses.

    Parses OS neighbor / ARP cache (Windows 'arp -a') for active entries on the local subnet.
    Filters out:
    - Gateway router(s) (Requirement 6)
    - Local server IP (Requirement 4)
    - Broadcast addresses (subnet broadcast, 255.255.255.255) (Requirement 7)
    - Multicast addresses (224.0.0.0/4) (Requirement 7)
    - Subnet network address (Requirement 7)
    - Loopback, link-local, unspecified, and non-unicast MACs
    Zero hardcoded IP addresses.
    """
    discovered = set()
    if profile is None:
        profile = get_active_network_profile()

    local_ip = profile.get("local_ip")
    local_subnet = profile.get("local_subnet")
    default_gw = profile.get("default_gateway")

    if not local_subnet:
        return discovered

    out = arp_output
    if out is None:
        try:
            import subprocess
            out = subprocess.check_output(["arp", "-a"], shell=False, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            out = ""

    if not out:
        return discovered

    current_iface_ip = None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Interface:"):
            # e.g. "Interface: 192.168.0.4 --- 0x11"
            parts = line.split()
            if len(parts) >= 2:
                current_iface_ip = parts[1]
            continue

        # If local_ip is known, only accept entries from the active interface's section
        if local_ip and current_iface_ip and current_iface_ip != local_ip:
            continue

        parts = line.split()
        if len(parts) >= 3:
            ip_str = parts[0]
            mac_str = parts[1].lower()
            entry_type = parts[2].lower()

            # Record into ARP MAC map for diagnostic Layer-2 lookup
            _arp_mac_map[ip_str] = mac_str

            try:
                ip_obj = _ipaddress.IPv4Address(ip_str)
            except Exception:
                continue

            # Exclude special/non-host addresses
            if (
                ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_unspecified
                or ip_str == "255.255.255.255"
            ):
                continue

            # Must belong to local monitored subnet
            if ip_obj not in local_subnet:
                continue

            # Exclude network identifier and subnet broadcast
            if ip_obj == local_subnet.network_address or ip_obj == local_subnet.broadcast_address:
                continue

            # Exclude server's own IP (it is added as HOST separately)
            if local_ip and ip_str == local_ip:
                continue

            # Exclude default gateway (Requirement 6)
            if default_gw and ip_str == default_gw:
                continue
            if ip_str in get_default_gateways():
                continue

            # Exclude broadcast/multicast MACs
            if mac_str in ("ff-ff-ff-ff-ff-ff", "00-00-00-00-00-00") or mac_str.startswith("01-00-5e"):
                continue

            # Accept dynamic neighbor entries or valid physical unicast MAC
            if entry_type == "dynamic" or ("-" in mac_str and not mac_str.startswith("01-")):
                discovered.add(ip_str)

    return discovered


_arp_mac_map: Dict[str, str] = {}


def get_mac_for_ip(ip_str: str) -> Optional[str]:
    """Look up Layer-2 MAC address for an IP from cache or system ARP table."""
    clean_ip = str(ip_str).strip()
    if clean_ip in _arp_mac_map:
        return _arp_mac_map[clean_ip]
    try:
        import subprocess
        out = subprocess.check_output(["arp", "-a"], shell=False, text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == clean_ip:
                mac = parts[1].replace("-", ":").lower()
                _arp_mac_map[clean_ip] = mac
                return mac
    except Exception:
        pass
    return None


def is_local_subnet_ip(ip: str) -> bool:
    """Return True if ip belongs to the monitored local subnet and is not gateway or special.

    Validates:
    - Valid IPv4
    - Belongs to dynamically discovered monitored local network
    - NOT default gateway
    - NOT network address or subnet broadcast
    - NOT multicast (224.0.0.0/4), loopback (127.0.0.0/8), link-local (169.254.0.0/16),
      unspecified (0.0.0.0), or general broadcast (255.255.255.255).
    """
    try:
        addr = _ipaddress.ip_address(ip)
        if addr.version != 4:
            return False
        addr_str = str(addr)
        if (
            addr.is_unspecified
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr_str == "255.255.255.255"
        ):
            return False

        gateways = get_default_gateways()
        if addr_str in gateways:
            return False

        monitored = get_monitored_local_networks()
        for net in monitored:
            if addr in net:
                if addr == net.network_address or addr == net.broadcast_address:
                    return False
                return True
        return False
    except ValueError:
        return False


def get_authorized_devices() -> list:
    """Return list of all currently authorized monitored devices."""
    with state_lock:
        return [copy.deepcopy(dev) for dev in _authorized_monitored_devices.values()]


def get_pending_requests() -> list:
    """Return list of all pending device access requests."""
    with state_lock:
        return [copy.deepcopy(req) for req in _pending_device_requests.values()]


def process_client_request(client_ip: str, device_token: Optional[str] = None) -> Optional[dict]:
    """Process incoming client web request and register pending request if unapproved."""
    global _server_local_ip
    srv_ip = get_server_local_ip()
    clean_ip = str(client_ip).strip()

    is_server_req = (srv_ip and clean_ip == srv_ip) or (clean_ip in ("127.0.0.1", "::1", "localhost"))

    with state_lock:
        if is_server_req:
            srv_key = "server-primary"
            # Invalidate any stale server entries with different keys or old IPs
            for k, dev in list(_authorized_monitored_devices.items()):
                if dev.get("is_server") and k != srv_key:
                    _authorized_monitored_devices.pop(k, None)

            if srv_key not in _authorized_monitored_devices:
                _authorized_monitored_devices[srv_key] = {
                    "device_id": srv_key,
                    "client_ip": srv_ip or "127.0.0.1",
                    "device_name": "CyberGraph Server",
                    "status": "AUTHORIZED",
                    "is_server": True,
                    "approval_timestamp": time.time(),
                    "last_seen_timestamp": time.time(),
                    "request_id": "srv-0",
                }
                if srv_ip:
                    _discovered_lan_hosts.add(srv_ip)
            else:
                _authorized_monitored_devices[srv_key]["last_seen_timestamp"] = time.time()
                if srv_ip:
                    _authorized_monitored_devices[srv_key]["client_ip"] = srv_ip
                    _discovered_lan_hosts.add(srv_ip)
            return _authorized_monitored_devices[srv_key]

        # Non-server clients MUST NEVER be assigned is_server=True
        if device_token and device_token in _authorized_monitored_devices:
            dev = _authorized_monitored_devices[device_token]
            if dev["status"] == "AUTHORIZED" and not dev.get("is_server"):
                dev["last_seen_timestamp"] = time.time()
                dev["client_ip"] = clean_ip
                if is_local_subnet_ip(clean_ip):
                    _discovered_lan_hosts.add(clean_ip)
                return dev

        for dev in _authorized_monitored_devices.values():
            if dev.get("status") == "AUTHORIZED" and dev.get("client_ip") == clean_ip and not dev.get("is_server"):
                dev["last_seen_timestamp"] = time.time()
                if is_local_subnet_ip(clean_ip):
                    _discovered_lan_hosts.add(clean_ip)
                return dev

        for req in _pending_device_requests.values():
            if req.get("client_ip") == clean_ip or (device_token and req.get("device_id") == device_token):
                req["timestamp"] = time.time()
                req["client_ip"] = clean_ip
                return req

        req_id = f"req-{uuid.uuid4().hex[:8]}"
        dev_id = device_token or f"dev-{uuid.uuid4().hex[:12]}"
        pending_obj = {
            "request_id": req_id,
            "device_id": dev_id,
            "client_ip": clean_ip,
            "status": "PENDING",
            "timestamp": time.time(),
            "device_name": f"LAN Device ({clean_ip})",
            "is_server": False,
        }
        _pending_device_requests[req_id] = pending_obj
        logger.info(f"[DEVICE ACCESS REQUEST] IP: {clean_ip} | Request ID: {req_id} | Status: PENDING")
        return pending_obj


def approve_device_request(identifier: str) -> Optional[dict]:
    """Approve a pending device access request, turning it into an AUTHORIZED HOST."""
    with state_lock:
        target_req = None
        target_key = None
        for k, req in list(_pending_device_requests.items()):
            if req.get("request_id") == identifier or req.get("device_id") == identifier or req.get("client_ip") == identifier:
                target_req = req
                target_key = k
                break

        if not target_req:
            clean_id = str(identifier).strip()
            for dev in _authorized_monitored_devices.values():
                if dev.get("device_id") == clean_id or dev.get("client_ip") == clean_id:
                    dev["status"] = "AUTHORIZED"
                    dev["approval_timestamp"] = time.time()
                    if is_local_subnet_ip(dev["client_ip"]):
                        _discovered_lan_hosts.add(dev["client_ip"])
                    return dev
            if is_local_subnet_ip(clean_id) or clean_id == get_server_local_ip():
                dev_id = f"dev-{uuid.uuid4().hex[:12]}"
                auth_obj = {
                    "device_id": dev_id,
                    "client_ip": clean_id,
                    "status": "AUTHORIZED",
                    "device_name": f"Approved Device ({clean_id})",
                    "approval_timestamp": time.time(),
                    "last_seen_timestamp": time.time(),
                    "is_server": (clean_id == get_server_local_ip()),
                    "request_id": f"manual-{uuid.uuid4().hex[:6]}",
                }
                _authorized_monitored_devices[dev_id] = auth_obj
                _discovered_lan_hosts.add(clean_id)
                logger.info(f"[EXPLICIT APPROVAL] Manually authorized IP: {clean_id}")
                return auth_obj
            return None

        _pending_device_requests.pop(target_key, None)
        dev_id = target_req["device_id"]
        auth_obj = {
            "device_id": dev_id,
            "client_ip": target_req["client_ip"],
            "status": "AUTHORIZED",
            "device_name": f"Approved Device ({target_req['client_ip']})",
            "approval_timestamp": time.time(),
            "last_seen_timestamp": time.time(),
            "is_server": False,
            "request_id": target_req["request_id"],
        }
        _authorized_monitored_devices[dev_id] = auth_obj
        if is_local_subnet_ip(auth_obj["client_ip"]):
            _discovered_lan_hosts.add(auth_obj["client_ip"])

        logger.info(f"[EXPLICIT APPROVAL] Device {dev_id} ({auth_obj['client_ip']}) is now AUTHORIZED_HOST")
        return auth_obj


def reject_device_request(identifier: str) -> bool:
    """Reject a pending device request."""
    with state_lock:
        for k, req in list(_pending_device_requests.items()):
            if req.get("request_id") == identifier or req.get("device_id") == identifier or req.get("client_ip") == identifier:
                _pending_device_requests.pop(k, None)
                logger.info(f"[DEVICE REJECTED] Access request {identifier} rejected")
                return True
        return False


def revoke_device_authorization(identifier: str) -> bool:
    """Revoke authorization for a device so it is no longer a HOST."""
    with state_lock:
        revoked = False
        for k, dev in list(_authorized_monitored_devices.items()):
            if dev.get("device_id") == identifier or dev.get("client_ip") == identifier:
                if not dev.get("is_server"):
                    cip = dev.get("client_ip")
                    _authorized_monitored_devices.pop(k, None)
                    if cip in _discovered_lan_hosts:
                        _discovered_lan_hosts.remove(cip)
                    logger.info(f"[AUTHORIZATION REVOKED] Device {identifier} ({cip}) revoked")
                    revoked = True
        return revoked


def resolve_node_type(ip: str, device_context: Optional[dict] = None) -> str:
    """Determine semantic node classification (HOST, GATEWAY, PENDING, EXTERNAL) for an IP address.

    Classification priority (per specification):
    1. AUTHORIZED MONITORED DEVICE → HOST (matched by token context or authorized client IP on active local network)
    2. CyberGraph server's own local IP → HOST
    3. Dynamically detected default gateway → GATEWAY
    4. Confirmed LAN neighbor that is explicitly authorized → HOST
    5. Pending/unapproved LAN device → PENDING
    6. Anything else observed in traffic → EXTERNAL

    CRITICAL INVARIANT:
    Private RFC1918 address alone MUST NEVER make an IP a HOST.
    """
    clean_ip = str(ip).replace("HOST:", "").replace("EXTERNAL:", "").replace("GATEWAY:", "").replace("EXT:", "").replace("GW:", "").replace("PENDING:", "").replace("SUSPICIOUS:", "").strip()

    # Priority 1: Explicitly Authorized Monitored Device
    if device_context and device_context.get("status") == "AUTHORIZED":
        return "HOST"

    # Check authorized devices registry by IP
    for dev in _authorized_monitored_devices.values():
        if dev.get("status") == "AUTHORIZED":
            dev_ip = dev.get("client_ip")
            if dev_ip and dev_ip == clean_ip:
                return "HOST"

    # Priority 2: CyberGraph server's own local IP
    server_ip = get_server_local_ip()
    if server_ip and clean_ip == server_ip:
        return "HOST"

    # Priority 3: Default Gateway
    if clean_ip in get_default_gateways():
        return "GATEWAY"

    # Priority 4: Check explicit _discovered_lan_hosts set
    if clean_ip in _discovered_lan_hosts:
        return "HOST"

    # Priority 5: Pending / unapproved LAN device
    if device_context and device_context.get("status") == "PENDING":
        return "PENDING"
    for req in _pending_device_requests.values():
        if req.get("client_ip") == clean_ip and req.get("status") == "PENDING":
            return "PENDING"

    # Priority 6: Anything else -> EXTERNAL
    return "EXTERNAL"


def format_endpoint_label(ip: str) -> str:
    """Format endpoint IP with its resolved type prefix for SOC alert & threat feed display."""
    clean_ip = str(ip).replace("HOST:", "").replace("EXTERNAL:", "").replace("GATEWAY:", "").replace("EXT:", "").replace("GW:", "").replace("PENDING:", "").replace("SUSPICIOUS:", "").strip()
    ntype = resolve_node_type(clean_ip)
    return f"{ntype}:{clean_ip}"


def is_local_lan_host(ip: str) -> bool:
    """Return True if ip is strictly a confirmed HOST endpoint on the monitored local LAN."""
    clean_ip = str(ip).replace("HOST:", "").replace("EXTERNAL:", "").replace("GATEWAY:", "").replace("EXT:", "").replace("GW:", "").replace("PENDING:", "").replace("SUSPICIOUS:", "").strip()
    return resolve_node_type(clean_ip) == "HOST"


# Backwards-compatibility aliases
is_private_ip = is_local_subnet_ip


def get_server_local_ip() -> Optional[str]:
    """Return the CyberGraph server's local IPv4 address."""
    global _server_local_ip
    if not _server_local_ip:
        prof = get_active_network_profile()
        _server_local_ip = prof.get("local_ip")
    return _server_local_ip


def set_server_local_ip(ip: Optional[str]):
    """Explicitly set or override server's local IP (for testing or manual config)."""
    global _server_local_ip
    _server_local_ip = str(ip) if ip else None
    with state_lock:
        # Invalidate any old server entry
        for k, dev in list(_authorized_monitored_devices.items()):
            if dev.get("is_server"):
                _authorized_monitored_devices.pop(k, None)
        if _server_local_ip:
            if is_local_subnet_ip(_server_local_ip):
                _discovered_lan_hosts.add(_server_local_ip)
            srv_key = "server-primary"
            _authorized_monitored_devices[srv_key] = {
                "device_id": srv_key,
                "client_ip": _server_local_ip,
                "device_name": "CyberGraph Server",
                "status": "AUTHORIZED",
                "is_server": True,
                "approval_timestamp": time.time(),
                "last_seen_timestamp": time.time(),
                "request_id": "srv-0",
            }


def add_discovered_lan_host(ip: str) -> bool:
    """Validate and register a discovered LAN host endpoint into persistent registry."""
    global _discovered_lan_hosts
    if is_local_subnet_ip(ip):
        _discovered_lan_hosts.add(str(ip).strip())
        return True
    return False


def get_discovered_lan_hosts() -> set:
    """Return copy of the set of persistent LAN hosts."""
    return set(_discovered_lan_hosts)


_active_network_profile_override: Optional[dict] = None


def set_network_profile_override(profile: Optional[dict]):
    """Set persistent network profile override (for testing or manual configuration)."""
    global _active_network_profile_override, _current_network_id
    _active_network_profile_override = profile
    _current_network_id = None
    sync_network_state(profile_override=profile)


def sync_network_state(force_scan: bool = False, profile_override: Optional[dict] = None):
    """Dynamically synchronize network state and handle network switches."""
    global _current_network_id, _server_local_ip, _discovered_lan_hosts
    global _monitored_local_networks, _default_gateways, _last_neighbor_scan_time
    global _active_network_profile_override

    if profile_override is not None:
        _active_network_profile_override = profile_override

    profile = _active_network_profile_override if _active_network_profile_override is not None else get_active_network_profile()
    iface = profile.get("iface")
    local_ip = profile.get("local_ip")
    subnet = profile.get("local_subnet")
    gateway = profile.get("default_gateway")

    network_id = (str(local_ip), str(subnet), str(gateway))
    network_changed = (_current_network_id is not None and network_id != _current_network_id)

    if _current_network_id is None or network_changed:
        if network_changed:
            logger.info(
                f"[NETWORK CHANGE DETECTED] Switching network: {_current_network_id} -> {network_id}"
            )
        # Clear old LAN host registry, non-server authorized devices, and old pending requests
        _discovered_lan_hosts.clear()
        _pending_device_requests.clear()
        for k, dev in list(_authorized_monitored_devices.items()):
            if not dev.get("is_server") or network_changed or _current_network_id is None:
                _authorized_monitored_devices.pop(k, None)

        # Update monitored networks
        _monitored_local_networks.clear()
        if subnet:
            if isinstance(subnet, _ipaddress.IPv4Network):
                _monitored_local_networks.add(subnet)
            else:
                _monitored_local_networks.add(_ipaddress.IPv4Network(str(subnet), strict=False))

        # Update default gateways
        _default_gateways.clear()
        if gateway:
            _default_gateways.add(str(gateway))

        if fw_executor:
            if local_ip:
                fw_executor.add_protected_ip(str(local_ip))
            if gateway:
                fw_executor.add_protected_ip(str(gateway))

        # Register server's own local IP as HOST (Requirement 4 & 12)
        _server_local_ip = str(local_ip) if local_ip else None
        if _server_local_ip:
            _discovered_lan_hosts.add(_server_local_ip)
            srv_key = "server-primary"
            _authorized_monitored_devices[srv_key] = {
                "device_id": srv_key,
                "client_ip": _server_local_ip,
                "device_name": "CyberGraph Server",
                "status": "AUTHORIZED",
                "is_server": True,
                "approval_timestamp": time.time(),
                "last_seen_timestamp": time.time(),
                "request_id": "srv-0",
            }
            # Clean up any duplicate/stale server entries
            for k, dev in list(_authorized_monitored_devices.items()):
                if dev.get("is_server") and (k != srv_key or dev.get("client_ip") != _server_local_ip):
                    _authorized_monitored_devices.pop(k, None)

        _current_network_id = network_id
        _last_neighbor_scan_time = time.time()

        logger.info(
            f"[NETWORK STATE SYNC] active_iface={iface} | local_ip={_server_local_ip} | "
            f"subnet={subnet} | gateway={gateway} | "
            f"authorized_monitored_hosts={len(_authorized_monitored_devices)}"
        )


def get_stream_pool():
    global df_stream_pool
    if df_stream_pool is None:
        try:
            df_full = load_dataset()
            df_stream_pool = df_full.iloc[int(len(df_full) * 0.75):].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not load offline stream pool: {e}")
            df_stream_pool = pd.DataFrame()
    return df_stream_pool


def _run_pipeline_cycle(batch_override=None):
    """Executes a single processing cycle for the server-authoritative pipeline."""
    global stream_cursor, shared_state, _last_pipeline_state, _last_explain_state, _recent_event_history
    global _discovered_lan_hosts

    init_engines()
    batch_sz = 15
    with state_lock:
        mode = shared_state["mode"]
        current_stats = copy.deepcopy(shared_state["stats"])

    sync_network_state()

    capture_available = is_live_capture_available()
    recent_live = get_recent_live_flows(window_seconds=3.0)
    has_recent_live = len(recent_live) > 0

    if batch_override is not None:
        data_source = "TEST_BATCH_OVERRIDE"
        buf_count = len(batch_override)
        batch_records = list(batch_override)
    elif mode == "offline":
        pool = get_stream_pool()
        data_source = "OFFLINE DATASET STREAM (NF-UNSW-NB15)"
        buf_count = 0
        if len(pool) > 0:
            if stream_cursor + batch_sz >= len(pool):
                stream_cursor = 0
            batch_df = pool.iloc[stream_cursor : stream_cursor + batch_sz]
            stream_cursor += batch_sz
            batch_records = batch_df.to_dict(orient="records")
        else:
            batch_records = []

    elif mode == "attack":
        pool = get_stream_pool()
        if capture_available:
            data_source = "LIVE WI-FI PACKETS + REPLAY ATTACK SCENARIO"
            buf_count = len(recent_live)
            attack_sub = pool[pool["label_enc"] == 1] if len(pool) > 0 and "label_enc" in pool.columns else pool
            batch_df = attack_sub.sample(n=min(batch_sz, len(attack_sub))) if len(attack_sub) > 0 else attack_sub
            batch_records = recent_live[-5:] + batch_df.to_dict(orient="records")[:max(1, batch_sz - len(recent_live[-5:]))]
        else:
            data_source = "REPLAY ATTACK SCENARIO (STATIC DATASET STREAM - WI-FI DISCONNECTED)"
            buf_count = 0
            attack_sub = pool[pool["label_enc"] == 1] if len(pool) > 0 and "label_enc" in pool.columns else pool
            batch_df = attack_sub.sample(n=min(batch_sz, len(attack_sub))) if len(attack_sub) > 0 else attack_sub
            batch_records = batch_df.to_dict(orient="records")

        now_ms = time.time() * 1000.0
        for r in batch_records:
            r["timestamp"] = now_ms

    else:  # mode == "normal"
        if capture_available and has_recent_live:
            data_source = "LIVE WI-FI PACKETS (PHYSICAL CAPTURE ACTIVE)"
            buf_count = len(recent_live)
            batch_records = recent_live[-batch_sz:]
        else:
            pool = get_stream_pool()
            data_source = "OFFLINE DATASET STREAM (NF-UNSW-NB15)"
            buf_count = len(recent_live)
            if len(pool) > 0:
                if stream_cursor + batch_sz >= len(pool):
                    stream_cursor = 0
                batch_df = pool.iloc[stream_cursor : stream_cursor + batch_sz]
                stream_cursor += batch_sz
                batch_records = batch_df.to_dict(orient="records")
            else:
                batch_records = []

    evals = []
    host_risk_map = {}
    host_nodes_dict = {}
    service_nodes_dict = {}
    raw_edge_list = []
    edge_risk_list = []
    risk_scores = []
    l1_count = 0
    l2_count = 0
    l3_count = 0
    host_risk_map = {}

    last_lat = {"capture": 1.25, "feature_prep": 0.65, "gnn_infer": 6.80, "total": 8.70}
    last_pipeline = {"xgb_prob": 0.001, "ae_score": 0.012, "fusion_score": 0.001, "gnn_risk": 0.01, "final_tier": "BENIGN"}

    for rec in batch_records:
        try:
            t_gnn_start = time.perf_counter()
            flow_rec = traffic_engine.process_flow_record(rec)

            # Local broadcast/multicast discovery protection
            flow_dst_ip = str(flow_rec.get("dst_ip", ""))
            is_lan_discovery = (
                flow_dst_ip == "255.255.255.255"
                or flow_dst_ip.endswith(".255")
                or flow_dst_ip.startswith("224.0.0.")
                or flow_dst_ip.startswith("239.255.")
            )
            if is_lan_discovery:
                # Local network broadcast/multicast (e.g. SSDP, NetBIOS, printer discovery) is normal background noise
                flow_rec["xgb_prob"] = min(float(flow_rec.get("xgb_prob", 0.0)), 0.05)
                flow_rec["fusion_score"] = min(float(flow_rec.get("fusion_score", 0.0)), 0.05)

            base_fusion = float(flow_rec["fusion_score"])
            gnn_edge_risk = 0.0
            pyg_data = None
            gnn_activation_reason = "NONE"
            combined_risk = base_fusion  # default: GNN not available / no edges yet

            if gnn_stream is not None:
                gnn_stream.ingest_flow(flow_rec)
                probs, pyg_data, h_map, s_map, edge_prob_map = gnn_stream.evaluate_realtime_gnn_risk()
                if len(probs) > 0:
                    edge_key = (
                        f"HOST:{flow_rec['src_ip']}",
                        f"SERVICE:{flow_rec['dst_port']}/{flow_rec['protocol']}"
                    )
                    gnn_edge_risk = edge_prob_map.get(edge_key, 0.0)
                    logger.debug(
    "[RISK RECONCILIATION] src_ip=%s dst_ip=%s dst_port=%s protocol=%s "
    "xgb=%.4f ae=%.4f fusion=%.4f raw_gnn=%.4f effective_gnn=%.4f",
    flow_rec.get("src_ip"),
    flow_rec.get("dst_ip"),
    flow_rec.get("dst_port"),
    flow_rec.get("protocol"),
    float(flow_rec.get("xgb_prob", 0.0)),
    float(flow_rec.get("ae_score", 0.0)),
    base_fusion,
    raw_gnn_edge_risk if "raw_gnn_edge_risk" in locals() else gnn_edge_risk,
    gnn_edge_risk,
)
                    base_fusion = float(flow_rec.get("fusion_score", 0.0))
                    xgb_prob = float(flow_rec.get("xgb_prob", 0.0))
                    ae_score = float(flow_rec.get("ae_score", 0.0))

# ── Guarded GNN evidence gate ─────────────────────────────────────────────
# AE anomaly alone (without any corroborating XGB/Fusion signal) must NOT
# unlock historical GNN edge risk. This prevents a single out-of-distribution
# ICMP/zero-port flow from inheriting a high accumulated GRU state and being
# misclassified as CRITICAL.
#
# Activation rules (first match wins):
#   XGB_STRONG       : xgb_prob >= 0.70
#   FUSION_STRONG    : base_fusion >= 0.35
#   AE_PLUS_CORROBORATION : ae_score >= 0.50 AND (xgb_prob >= 0.40 OR base_fusion >= 0.20)
#   NONE             : no sufficient evidence → GNN cannot dominate
                    if xgb_prob >= 0.70:
                        gnn_activation_reason = "XGB_STRONG"
                        base_ml_evidence = True
                    elif base_fusion >= 0.35:
                        gnn_activation_reason = "FUSION_STRONG"
                        base_ml_evidence = True
                    elif ae_score >= 0.50 and (xgb_prob >= 0.40 or base_fusion >= 0.20):
                        gnn_activation_reason = "AE_PLUS_CORROBORATION"
                        base_ml_evidence = True
                    else:
                        gnn_activation_reason = "NONE"
                        base_ml_evidence = False

# GNN can dominate combined_risk only when base evidence is confirmed.
                    if base_ml_evidence:
                        combined_risk = max(base_fusion, gnn_edge_risk)
                    else:
                        combined_risk = base_fusion
            t_gnn_end = time.perf_counter()
            gnn_infer_ms = (t_gnn_end - t_gnn_start) * 1000.0
        except Exception as _rec_exc:
            logger.warning("Skipping malformed flow record: %s", _rec_exc)
            continue

        if "latency_breakdown" in flow_rec:
            last_lat = {
                "capture": flow_rec["latency_breakdown"]["capture_latency_ms"],
                "feature_prep": flow_rec["latency_breakdown"]["feature_prep_ms"],
                "gnn_infer": round(gnn_infer_ms, 2),
                "total": round(flow_rec["latency_breakdown"]["capture_latency_ms"] + flow_rec["latency_breakdown"]["feature_prep_ms"] + gnn_infer_ms, 2),
            }

        risk_scores.append(combined_risk)

        _bari = bari_engine.score(
            src_ip=flow_rec["src_ip"],
            dst_port=flow_rec["dst_port"],
            protocol=flow_rec["protocol"],
            gnn_risk=combined_risk,
            timestamp=flow_rec["timestamp"],
        )

        attack_type = _classify_attack_type(flow_rec)
        correlation_engine.record_event(
            src_ip=flow_rec["src_ip"],
            dst_ip=flow_rec["dst_ip"],
            dst_port=flow_rec["dst_port"],
            protocol=flow_rec["protocol"],
            attack_type=attack_type,
            gnn_risk=combined_risk,
            xgb_prob=float(flow_rec["xgb_prob"]),
            ae_score=float(flow_rec["ae_score"]),
            fusion_score=float(flow_rec["fusion_score"]),
            risk_level="L3" if combined_risk >= RISK_L2_MAX else ("L2" if combined_risk >= RISK_L1_MAX else "L1"),
            geo=_lookup_geo(flow_rec["src_ip"]),
            capture_ms=last_lat["capture"],
            feature_prep_ms=last_lat["feature_prep"],
            gnn_infer_ms=last_lat["gnn_infer"],
        )
        logger.info(
    "[FINAL RISK DEBUG] "
    "src_ip=%s dst_ip=%s src_port=%s dst_port=%s protocol=%s "
    "xgb=%.4f ae=%.4f fusion=%.4f gnn=%.4f combined=%.4f bari=%.4f "
    "gnn_activation=%s",
    flow_rec["src_ip"],
    flow_rec.get("dst_ip", "?"),
    flow_rec.get("src_port", flow_rec.get("_src_port", 0)),
    flow_rec["dst_port"],
    flow_rec.get("protocol", "?"),
    float(flow_rec["xgb_prob"]),
    float(flow_rec["ae_score"]),
    float(flow_rec["fusion_score"]),
    float(gnn_edge_risk),
    float(combined_risk),
    float(_bari.get("bari_score", 0.0)),
    gnn_activation_reason,
)
        eval_res = alert_engine.process_flow_eval(
            flow_rec, risk_score=combined_risk, snapshot=pyg_data,
            xgb_model=traffic_engine.xgb_model, bari_res=_bari,
            node_type_resolver=resolve_node_type
        )

        attack_journey_payload = eval_res.get("attack_journey", {})
        correlation_report = correlation_engine._build_report(flow_rec["src_ip"])
        mitre_mappings = mitre_mapper.map_flow_to_techniques(
            flow_record=flow_rec,
            bari_result=_bari,
            correlation_result=correlation_report,
            attack_journey=attack_journey_payload,
        )
        eval_res["mitre_mappings"] = mitre_mappings

        _last_pipeline_state = {
            "xgb_prob": float(flow_rec["xgb_prob"]),
            "ae_score": float(flow_rec["ae_score"]),
            "fusion_score": base_fusion,
            "gnn_risk": combined_risk,
            "final_tier":  (
    "CRITICAL_ATTACK_THREAT"
    if combined_risk >= RISK_L2_MAX
    else "EARLY_WARNING_THREAT"
    if combined_risk >= RISK_L1_MAX
    else "BENIGN"
),
            "bari_score": float(_bari.get("bari_score", 0.0)),
            "mitre_mappings_count": len(mitre_mappings),
        }
        if combined_risk >= RISK_L2_MAX:
            display_tier = "CRITICAL_ATTACK_THREAT"
        elif combined_risk >= RISK_L1_MAX:
            display_tier = "EARLY_WARNING_THREAT"
        else:
            display_tier = "BENIGN"
        raw_feat = flow_rec.get("raw_features", None)
        if raw_feat is not None and len(raw_feat) > 0:
            _last_explain_state = {
                "edge_attr": [
                    flow_rec["timestamp"], flow_rec["dst_port"], flow_rec["protocol"],
                    flow_rec["byte_count"], flow_rec["pkt_count"], flow_rec["duration"],
                    flow_rec["xgb_prob"], flow_rec["ae_score"], flow_rec["fusion_score"],
                ],
                "risk_score": combined_risk,
                "source_host": eval_res["source_host"],
                "target_service": eval_res["target_service"],
                "raw_feat_1d": raw_feat,
                "xgb_model": traffic_engine.xgb_model,
            }

        if "Level 1" in eval_res["level"]:
            l1_count += 1
        elif "Level 2" in eval_res["level"]:
            l2_count += 1
        else:
            l3_count += 1

        clean_src = str(flow_rec["src_ip"]).strip()
        dst_s = eval_res["target_service"]
        clean_dst = dst_s.replace("SERVICE:", "")

        # Format source_host in alert/eval res based on resolved node type (HOST:ip, GATEWAY:ip, EXTERNAL:ip)
        correct_source_host = format_endpoint_label(clean_src)
        old_sh = eval_res.get("source_host", "")
        eval_res["source_host"] = correct_source_host
        if "summary" in eval_res and old_sh in eval_res["summary"]:
            eval_res["summary"] = eval_res["summary"].replace(old_sh, correct_source_host)

        # Diagnostic check for UNCONFIRMED LAN ENDPOINT:
        # 1. belongs to current local subnet
        # 2. is NOT the server IP
        # 3. is NOT the gateway
        # 4. is NOT a confirmed LAN neighbor
        is_subnet_ip = is_local_subnet_ip(clean_src)
        is_server = (clean_src == get_server_local_ip())
        is_gw = (clean_src in get_default_gateways())
        is_confirmed_peer = (clean_src in get_discovered_lan_hosts())

        if is_subnet_ip and not is_server and not is_gw and not is_confirmed_peer:
            raw_src_mac = (
                flow_rec.get("src_mac")
                or rec.get("_src_mac")
                or rec.get("src_mac")
                or get_mac_for_ip(clean_src)
                or "UNKNOWN"
            )
            src_mac_val = str(raw_src_mac).replace(":", "-").upper() if raw_src_mac != "UNKNOWN" else "UNKNOWN"
            dst_ip_val = flow_rec.get("dst_ip", rec.get("_dst_ip", rec.get("IPV4_DST_ADDR", "UNKNOWN")))
            dst_port_val = flow_rec.get("dst_port", rec.get("_dst_port", rec.get("L4_DST_PORT", "UNKNOWN")))
            proto_val = flow_rec.get("protocol", rec.get("PROTOCOL", 0))
            proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
            proto_str = proto_map.get(proto_val, str(proto_val))
            iface_val = (
                flow_rec.get("iface")
                or rec.get("_iface")
                or rec.get("iface")
                or get_active_network_profile().get("iface")
            )

            diag_lines = [
                "[UNCONFIRMED LAN ENDPOINT]",
                f"src_ip={clean_src}",
                f"src_mac={src_mac_val}",
                f"dst_ip={dst_ip_val}",
                f"dst_port={dst_port_val}",
                f"protocol={proto_str}",
            ]
            if iface_val:
                diag_lines.append(f"iface={iface_val}")
            diag_lines.append("classification=EXTERNAL")
            logger.info("\n".join(diag_lines))

        # Track max risk per host for per-host coloring
        prev_r = host_risk_map.get(clean_src, 0.0)
        host_risk_map[clean_src] = max(prev_r, float(combined_risk))

        host_nodes_dict[clean_src] = clean_src
        service_nodes_dict[clean_dst] = clean_dst

        edge_level = "NORMAL"
        edge_color = "normal"
        if combined_risk >= RISK_L2_MAX:
            edge_level = "CRITICAL"
            edge_color = "critical"
        elif combined_risk >= RISK_L1_MAX:
            edge_level = "SUSPICIOUS"
            edge_color = "suspicious"

        # Edge source_label must match the graph node label so the frontend
        # can resolve nodePosMap[source_label] correctly.
        src_node_type = resolve_node_type(clean_src)
        if src_node_type == "HOST":
            edge_source_label = clean_src
        elif src_node_type == "GATEWAY":
            edge_source_label = f"GW:{clean_src}"
        elif src_node_type == "PENDING":
            edge_source_label = f"PENDING:{clean_src}"
        else:
            edge_source_label = f"EXT:{clean_src}"

        raw_edge_list.append({
            "source_label": edge_source_label,
            "target_label": clean_dst,
            "risk": combined_risk,
            "level": edge_level,
            "color": edge_color,
            "src_ip": clean_src,
            "dst_port": flow_rec.get("dst_port"),
            "protocol": flow_rec.get("protocol"),
        })

        evals.append(eval_res)
        with state_lock:
            _recent_event_history.append(copy.deepcopy(eval_res))
            if len(_recent_event_history) > 100:
                _recent_event_history.pop(0)

        last_pipeline = {
            "xgb_prob": float(flow_rec.get("xgb_prob", 0.001)),
            "ae_score": float(flow_rec.get("ae_score", 0.012)),
            "fusion_score": base_fusion,
            "gnn_risk": combined_risk,
            "final_tier": display_tier,
        }

    # Aggregate repeated connections from the same system to the same service (e.g. port 5000),
    # preserving source, destination, protocol, and service differences while grouping duplicates.
    aggregated_edges = {}
    for edge in raw_edge_list:
        pair_key = (edge["source_label"], edge["target_label"])
        if pair_key not in aggregated_edges:
            aggregated_edges[pair_key] = {
                "source_label": edge["source_label"],
                "target_label": edge["target_label"],
                "risk": edge["risk"],
                "level": edge["level"],
                "color": edge["color"],
                "count": 1,
                "dst_port": edge.get("dst_port"),
                "protocol": edge.get("protocol"),
            }
        else:
            entry = aggregated_edges[pair_key]
            entry["count"] += 1
            if edge["risk"] > entry["risk"]:
                entry["risk"] = edge["risk"]
                entry["level"] = edge["level"]
                entry["color"] = edge["color"]

    edge_risk_list = list(aggregated_edges.values())

    # Diagnostic logging before graph rendering
    srv_ip_log = get_server_local_ip()
    gw_log = sorted(list(get_default_gateways()))
    conf_hosts_log = sorted(list(get_discovered_lan_hosts()))
    for flow_endpoint in host_nodes_dict.keys():
        r_type = resolve_node_type(flow_endpoint)
        logger.info(
            f"[NODE CLASSIFICATION DIAGNOSTIC] "
            f"server_local_ip={srv_ip_log} | "
            f"gateway={gw_log} | "
            f"confirmed_lan_hosts={conf_hosts_log} | "
            f"flow_endpoint={flow_endpoint} | "
            f"resolved_node_type={r_type}"
        )

    # Build Graph Nodes.
    # HOSTs = ALL authorized monitored devices for active network (MUST remain persistent across zero traffic).
    # GATEWAY = active default gateway router node (distinct GATEWAY type).
    # PENDING = unapproved LAN clients requesting access (distinct PENDING type).
    # EXTERNALs = non-authorized / external endpoints seen in current batch (dynamic, not persistent).
    graph_nodes = []

    # --- Persistent HOST nodes (all authorized monitored devices for active network) ---
    authorized_hosts_set = set()
    srv_ip_val = get_server_local_ip()
    if srv_ip_val:
        authorized_hosts_set.add(srv_ip_val)

    with state_lock:
        for dev in _authorized_monitored_devices.values():
            if dev.get("status") == "AUTHORIZED":
                cip = dev.get("client_ip")
                if cip:
                    authorized_hosts_set.add(cip)
        for h in _discovered_lan_hosts:
            authorized_hosts_set.add(h)

    for host_ip in sorted(list(authorized_hosts_set)):
        h_risk = host_risk_map.get(host_ip, 0.0)

        if h_risk >= RISK_L2_MAX:
            t_level = "CRITICAL"
            t_color = "critical"
        elif h_risk >= RISK_L1_MAX:
            t_level = "SUSPICIOUS"
            t_color = "suspicious"
        else:
            t_level = "NORMAL"
            t_color = "normal"

        graph_nodes.append({
            "label": host_ip,       # e.g. "10.160.36.186"
            "type": "HOST",
            "threat_level": t_level,
            "color": t_color,
            "max_risk": h_risk,
        })

    # --- GATEWAY, PENDING & EXTERNAL nodes: non-HOST endpoints from current batch only (dynamic) ---
    for host_label in host_nodes_dict.values():
        n_type = resolve_node_type(host_label)
        if n_type != "HOST":
            h_risk = host_risk_map.get(host_label, 0.0)
            if h_risk >= RISK_L2_MAX:
                t_level = "CRITICAL"
                t_color = "critical"
            elif h_risk >= RISK_L1_MAX:
                t_level = "SUSPICIOUS"
                t_color = "suspicious"
            else:
                t_level = "SUSPICIOUS" if n_type == "PENDING" else "NORMAL"
                t_color = "suspicious" if n_type == "PENDING" else "normal"

            if n_type == "GATEWAY":
                # Default Gateway router node (Requirement: GATEWAY, distinct styling, not persistent HOST, not EXTERNAL)
                graph_nodes.append({
                    "label": f"GW:{host_label}",   # e.g. "GW:192.168.0.1"
                    "type": "GATEWAY",
                    "threat_level": t_level,
                    "color": "gateway",
                    "max_risk": h_risk,
                })
            elif n_type == "PENDING":
                # Pending unapproved LAN device (Requirement: PENDING / SUSPICIOUS styling)
                graph_nodes.append({
                    "label": f"PENDING:{host_label}",  # e.g. "PENDING:10.160.36.186"
                    "type": "PENDING",
                    "threat_level": t_level,
                    "color": "suspicious",
                    "max_risk": h_risk,
                })
            else:
                # External endpoint (e.g. 192.168.0.6, 192.168.0.7, 8.8.8.8)
                ext_color = t_color if t_color != "normal" else "external"
                graph_nodes.append({
                    "label": f"EXT:{host_label}",  # e.g. "EXT:192.168.0.6"
                    "type": "EXTERNAL",
                    "threat_level": t_level,
                    "color": ext_color,
                    "max_risk": h_risk,
                })

    for svc_label in service_nodes_dict.values():
        graph_nodes.append({
            "label": svc_label,
            "type": "SERVICE",
            "threat_level": "NORMAL",
            "color": "service",
            "max_risk": 0.0,
        })

    # Event Deduplication logic for threat feed
    deduped_evals = []
    seen_events = set()

    for ev in evals:
        event_key = (ev["source_host"], ev["target_service"], ev["level"])

        if event_key not in seen_events:
            seen_events.add(event_key)
            deduped_evals.append(ev)

    top_correlated = correlation_engine.get_top_correlated(top_n=5)
    avg_risk = float(np.mean(risk_scores)) if risk_scores else 0.05
    max_r = float(np.max(risk_scores)) if risk_scores else 0.05

    with state_lock:
        s = shared_state["stats"]
        s["total_flows"] += len(batch_records)
        s["level1_normal"] = l1_count
        s["level2_suspicious"] = l2_count
        s["level3_critical"] = l3_count
        s["active_hosts"] = len(host_nodes_dict)
        s["active_services"] = len(service_nodes_dict)
        s["current_data_source"] = data_source
        s["live_buffer_count"] = buf_count
        s["current_risk"] = avg_risk
        s["peak_risk"] = max(s["peak_risk"], max_r)
        s["risk_history"].append(avg_risk)
        if len(s["risk_history"]) > 30:
            s["risk_history"].pop(0)
        s["last_latency_ms"] = last_lat

        shared_state["latencies"] = last_lat
        shared_state["pipeline_metrics"] = last_pipeline
        shared_state["graph_nodes"] = graph_nodes
        shared_state["graph_edges"] = edge_risk_list[:25]
        shared_state["correlation_intel"] = top_correlated
        shared_state["server_status"]["last_processing_time"] = time.time()
        shared_state["server_status"]["capture_active"] = capture_available
        shared_state["server_status"]["capture_available"] = capture_available
        shared_state["server_status"]["traffic_recent"] = has_recent_live
        shared_state["server_status"]["processing_thread_alive"] = True

        # Retain threat cards for specified durations without resetting unrelated cards:
        # Level 1: 15 seconds (10-20s window)
        # Level 2: 45 seconds
        # Level 3: 75 seconds (60-90s window)
        now_ts = time.time()
        for ev in deduped_evals:
            lvl = ev.get("level", "")
            if "Level 3" in lvl or ev.get("risk_category") == "CRITICAL_ATTACK_THREAT":
                card_duration = 75.0
            elif "Level 2" in lvl or ev.get("risk_category") == "EARLY_WARNING_THREAT":
                card_duration = 45.0
            else:
                card_duration = 15.0

            card_key = (ev.get("source_host"), ev.get("target_service"), lvl)
            if card_key in _active_threat_cards:
                existing = _active_threat_cards[card_key]
                existing["eval"] = ev
                existing["expires_at"] = max(existing["expires_at"], now_ts + card_duration)
            else:
                _active_threat_cards[card_key] = {
                    "eval": ev,
                    "expires_at": now_ts + card_duration,
                    "created_at": now_ts,
                }

        # Evict expired threat cards
        expired_keys = [k for k, v in _active_threat_cards.items() if now_ts >= v["expires_at"]]
        for k in expired_keys:
            del _active_threat_cards[k]

        def _card_sort_key(item):
            lvl_str = item["eval"].get("level", "")
            sev = 3 if "Level 3" in lvl_str else (2 if "Level 2" in lvl_str else 1)
            return (sev, item["created_at"])

        sorted_cards = sorted(_active_threat_cards.values(), key=_card_sort_key, reverse=True)
        shared_state["evals"] = [c["eval"] for c in sorted_cards[:30]]

        shared_state["state_version"] += 1
        curr_ver = shared_state["state_version"]

    logger.info(
        f"[NIDS PIPELINE v{curr_ver}] mode={mode} | flows={len(batch_records)} | "
        f"max_risk={max_r:.4f} | L1={l1_count} L2={l2_count} L3={l3_count}"
    )

    return evals, graph_nodes, edge_risk_list, last_pipeline


def _pipeline_worker():
    """Background worker executing the global NIDS pipeline independently of HTTP clients."""
    time.sleep(1.0)
    while True:
        try:
            _run_pipeline_cycle()
            with state_lock:
                shared_state["server_status"]["processing_thread_alive"] = True
                shared_state["server_status"]["last_processing_time"] = time.time()
        except Exception as exc:
            logger.exception("Error in global pipeline worker loop: %s", exc)
            with state_lock:
                shared_state["server_status"]["last_error"] = str(exc)
        time.sleep(0.75)


@app.route("/")
def index():
    init_engines()
    is_admin = is_admin_request()
    client_ip = request.remote_addr or "127.0.0.1"
    token_to_render = ADMIN_TOKEN if client_ip in ("127.0.0.1", "::1", "localhost") else ""
    return render_template_string(HTML_TEMPLATE, admin_token=token_to_render, is_admin=is_admin)


@app.route("/api/set_mode", methods=["POST"])
def set_mode():
    init_engines()
    if not is_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized set_mode attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required to change operating mode"
        }), 403

    m = request.args.get("mode")
    if not m and request.is_json:
        data = request.get_json(silent=True) or {}
        m = data.get("mode")
    if not m or m not in ("normal", "attack", "offline"):
        m = "normal"
    with state_lock:
        shared_state["mode"] = m
        shared_state["state_version"] += 1
        v = shared_state["state_version"]
    logger.info(f"[GLOBAL MODE CHANGE] Server mode set to '{m}' (state_version={v})")
    try:
        _run_pipeline_cycle()
    except Exception as exc:
        logger.warning(f"Immediate cycle execution warning on mode change: {exc}")
    return jsonify({"status": "SUCCESS", "mode": m, "state_version": v})


@app.route("/api/stream_step")
def stream_step():
    """Returns a read-only snapshot of the shared server state."""
    init_engines()
    with state_lock:
        snapshot = copy.deepcopy(shared_state)
    return jsonify(snapshot)


@app.route("/api/pipeline_diag")
def pipeline_diag():
    """Returns the last 5 processed live flows with full diagnostic breakdown.
    Used to numerically validate that the ML pipeline is scoring correctly.
    """
    init_engines()
    from modules.live_traffic_engine import _diag_flows
    return jsonify({
        "description": "Last 5 live flows processed through the ML pipeline",
        "note": "ae_raw_mse should be < ae_mse_max=53.2 for normal traffic; xgb should be < 0.35 for normal",
        "ae_mse_max": 53.19590666580561,
        "ae_threshold": 0.08100921690464065,
        "flows": list(_diag_flows),
        "flow_count": len(_diag_flows),
    })


@app.route("/api/graph_intelligence")
def graph_intelligence():
    """Returns correlation engine data + Kill Chain stage analysis."""
    init_engines()
    top_correlated = correlation_engine.get_top_correlated(top_n=10)
    return jsonify({
        "correlated_ips": top_correlated,
        "kill_chain_stages": list(correlation_engine._timelines.keys()) if hasattr(correlation_engine, '_timelines') else [],
        "escalation_count": sum(1 for ip in correlation_engine._timelines if correlation_engine._build_report(ip).get("escalation_detected", False))
    })


@app.route("/api/pipeline_explain")
def pipeline_explain():
    """Returns the last flow's pipeline step breakdown and SHAP explanation."""
    init_engines()
    state = dict(_last_pipeline_state)
    explain_state = dict(_last_explain_state)

    xai_report = None
    if explain_state and "raw_feat_1d" in explain_state:
        try:
            xai_report = xai_engine.explain_flow_edge(
                edge_attr=explain_state["edge_attr"],
                risk_score=explain_state["risk_score"],
                source_host=explain_state["source_host"],
                target_service=explain_state["target_service"],
                raw_feat_1d=explain_state["raw_feat_1d"],
                xgb_model=explain_state["xgb_model"],
                feature_cols=None,
            )
        except Exception:
            pass

    mitre_explanations = []
    if state.get("mitre_mappings_count", 0) > 0:
        mitre_explanations.append(f"{state['mitre_mappings_count']} MITRE ATT&CK technique(s) mapped from behavioral evidence")

    return jsonify({
        "pipeline_state": state,
        "xai_explanation": xai_report,
        "mitre_explanations": mitre_explanations,
        "bari_context": {
            "route_string": "N/A (per-flow data)",
            "status": "BARI contextual enrichment active",
        }
    })


@app.route("/api/attack_journey/<src_ip>")
def attack_journey(src_ip):
    """Returns the multi-hop attack journey for a specific source IP."""
    init_engines()
    timeline = correlation_engine.get_timeline(src_ip)
    journey = attack_journey_engine._journeys.get(src_ip)
    top_priority = attack_journey_engine.get_top_threat_priorities(limit=1)

    if journey and journey["timeline"]:
        return jsonify({
            "source_ip": src_ip,
            "journey_path": journey["journey_path"],
            "route_string": journey["route_string"],
            "timeline": journey["timeline"][-10:],
            "attack_objective": journey.get("timeline", [{}])[-1].get("stage", "Unknown"),
            "threat_priority_score": top_priority[0]["threat_priority_score"] if top_priority else 0,
            "threat_priority_tier": top_priority[0]["threat_priority_tier"] if top_priority else "LOW",
            "detection_lead_time_sec": journey.get("detection_lead_time_sec", 0),
            "early_detection_time": journey.get("early_t"),
            "critical_detection_time": journey.get("critical_t"),
        })
    else:
        return jsonify({
            "source_ip": src_ip,
            "message": "No attack journey data found for this IP.",
            "timeline": timeline,
        })


@app.route("/api/threat_priorities")
def threat_priorities():
    """Returns ranked threat priority queue from Attack Journey Engine."""
    init_engines()
    priorities = attack_journey_engine.get_top_threat_priorities(limit=10)
    for p in priorities:
        src_ip = p["source_ip"]
        correlation_report = correlation_engine._build_report(src_ip)
        mitre_mappings = mitre_mapper.map_flow_to_techniques(
            flow_record={"src_ip": src_ip},
            bari_result={},
            correlation_result=correlation_report,
            attack_journey={"journey_path": [], "attack_objective": {}}
        )
        p["mitre_techniques"] = [
            {
                "technique_id": m["technique_id"],
                "technique_name": m["technique_name"],
                "confidence": m["confidence"]
            }
            for m in mitre_mappings
        ]
    return jsonify({
        "priorities": priorities,
        "mitre_supported": ["T1046", "T1110", "T1498"],
        "total_active_threats": len(priorities),
    })


@app.route("/api/simulate_attack_path")
def simulate_attack_path():
    """Defensive 'What-If' Attack Path Simulator."""
    init_engines()
    start_host = request.args.get("host", "10.0.0.21")
    target_port = int(request.args.get("port", 3306))
    strategy = request.args.get("strategy", "lateral_movement")
    sim = attack_journey_engine.simulate_attack_path(
        start_host=start_host, target_port=target_port,
        strategy=strategy, bari_engine=bari_engine
    )
    return jsonify(sim)


@app.route("/api/mitre_mappings")
def mitre_mappings():
    """Returns all supported MITRE ATT&CK technique mappings and metadata."""
    init_engines()
    supported = []
    for tech_id in mitre_mapper.get_supported_techniques():
        info = mitre_mapper.get_technique_info(tech_id)
        supported.append({
            "technique_id": info.technique_id,
            "technique_name": info.technique_name,
            "tactic": info.tactic,
            "description": info.description,
            "required_confidence_threshold": info.required_confidence_threshold,
            "evidence_weights": info.evidence_weights,
            "behavioral_indicators": info.behavioral_indicators,
        })
    return jsonify({
        "supported_techniques": supported,
        "unsupported_count": 17,
        "mapping_method": "evidence-based deterministic rules",
        "confidence_formula": "confidence = weighted evidence match × model confidence damping",
        "separation_of_concerns": "MITRE provides ATT&CK contextualization; XGBoost/AE/GNN/BARI detect and score behavior",
    })


@app.route("/api/export_report")
def export_report():
    init_engines()
    with state_lock:
        st = copy.deepcopy(shared_state["stats"])
        recent_events = copy.deepcopy(_recent_event_history)
        active_evals = copy.deepcopy(shared_state.get("evals", []))
        sim_blocked = list(fw_executor.simulated_acl_blocked_ips) if fw_executor else []
        real_blocked = list(fw_executor.active_os_firewall_rules.keys()) if fw_executor else []
        fw_history = list(fw_executor.action_history) if fw_executor and hasattr(fw_executor, "action_history") else []
        is_real_enabled = fw_executor.enable_real_blocking if fw_executor else False

    # Consolidate unique threat and audit events
    threat_records = []
    seen_ids = set()

    all_raw_events = recent_events + active_evals
    for ev in all_raw_events:
        rec_id = (ev.get("source_host"), ev.get("target_service"), ev.get("timestamp"))
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            threat_records.append(ev)

    # Sort descending by risk score and timestamp
    threat_records.sort(key=lambda x: (x.get("risk_score", 0.0), x.get("timestamp", 0.0)), reverse=True)

    # Format check for JSON export
    if request.args.get("format") == "json":
        return jsonify({
            "status": "SUCCESS",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "telemetry_summary": st,
            "firewall_status": {
                "real_blocking_enabled": is_real_enabled,
                "simulated_acl_blocked_ips": sim_blocked,
                "real_os_firewall_blocked_ips": real_blocked,
                "total_blocked_count": len(set(sim_blocked + real_blocked)),
                "action_history": fw_history,
            },
            "detected_threats_count": len(threat_records),
            "threat_events": threat_records,
        })

    # Build HTML Incident Rows
    threat_rows_html = ""
    if threat_records:
        for idx, ev in enumerate(threat_records[:50], 1):
            ts_val = ev.get("timestamp")
            if ts_val:
                try:
                    ts_float = float(ts_val)
                    if ts_float > 1e11:  # milliseconds
                        ts_float /= 1000.0
                    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_float))
                except Exception:
                    ts_str = str(ts_val)
            else:
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S")

            lvl = ev.get("level", "Level 1 — Normal")
            badge_color = ev.get("badge_color", "#10b981")
            risk_score = float(ev.get("risk_score", 0.0))
            src = ev.get("source_host", "UNKNOWN")
            dst = ev.get("target_service", "UNKNOWN")
            summary = ev.get("summary", "Normal Traffic Flow")

            # Evidence & Model Findings
            xai = ev.get("xai_explanation", {})
            reasons = xai.get("why_this_attack_reasons", []) if isinstance(xai, dict) else []
            reasons_html = "".join([f"<li>{r}</li>" for r in reasons]) if reasons else "<li>Baseline traffic profile match</li>"

            # MITRE mappings
            mitre_list = ev.get("mitre_mappings", [])
            mitre_html = ""
            if mitre_list:
                mitre_badges = []
                for m in mitre_list:
                    mitre_badges.append(f"<span style='background:#e0f2fe; color:#0369a1; padding:2px 6px; border-radius:4px; font-weight:700;'>{m.get('technique_id')} ({m.get('technique_name')}) [{(float(m.get('confidence',0))*100):.1f}%]</span>")
                mitre_html = f"<div style='margin-top:4px;'><strong>MITRE ATT&CK:</strong> {' '.join(mitre_badges)}</div>"

            # Recommended Action
            rec_fw = ev.get("recommended_firewall_action", {})
            target_ip = rec_fw.get("target_ip", src.replace("HOST:", "").replace("EXT:", "").replace("GW:", "").replace("PENDING:", ""))
            cmd_preview = rec_fw.get("windows_netsh_rule_preview", f"netsh advfirewall firewall add rule name=\"Block-GNN-{target_ip}\" dir=in action=block remoteip={target_ip}")
            rec_action = f"Block IP {target_ip} via OS Firewall Rule<br><code style='background:#f1f5f9; padding:2px 4px; font-size:11px;'>{cmd_preview}</code>" if risk_score >= 0.50 else "Monitor traffic; no blocking action required."

            # Blocking Status
            is_ip_blocked = (target_ip in sim_blocked) or (target_ip in real_blocked)
            block_status_html = f"<span style='color:#ef4444; font-weight:700;'>BLOCKED</span>" if is_ip_blocked else "<span style='color:#10b981; font-weight:700;'>ALLOWED / MONITORING</span>"

            threat_rows_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-family: monospace; font-size: 11px;">{ts_str}</td>
                <td style="padding: 10px;"><span style="background:{badge_color}; color:#fff; padding:3px 8px; border-radius:4px; font-weight:700; font-size:11px;">{lvl}</span><br><small>Risk: {risk_score:.4f}</small></td>
                <td style="padding: 10px;"><strong>Src:</strong> {src}<br><strong>Dst:</strong> {dst}</td>
                <td style="padding: 10px; font-size: 12px;">
                    <div>{summary}</div>
                    <ul style="margin:4px 0 0 16px; padding:0; font-size:11px; color:#475569;">{reasons_html}</ul>
                    {mitre_html}
                </td>
                <td style="padding: 10px; font-size: 11px;">{rec_action}</td>
                <td style="padding: 10px; font-size: 11px;">{block_status_html}</td>
            </tr>
            """
    else:
        threat_rows_html = "<tr><td colspan='6' style='text-align:center; padding:20px; color:#64748b;'>No security incidents recorded in current session.</td></tr>"

    # Build Firewall Action History Rows
    fw_rows_html = ""
    if fw_history:
        for act in reversed(fw_history[-30:]):
            act_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(act.get("timestamp", time.time())))
            act_type = act.get("action", "UNKNOWN")
            act_ip = act.get("ip", "-")
            act_mode = act.get("execution_mode", "SIMULATED_MEMORY_ONLY")
            act_status = act.get("status", "-")
            fw_rows_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 8px; font-family: monospace; font-size: 11px;">{act_ts}</td>
                <td style="padding: 8px; font-weight:700; color:{'#ef4444' if act_type == 'BLOCK' else '#10b981'};">{act_type}</td>
                <td style="padding: 8px; font-family: monospace;">{act_ip}</td>
                <td style="padding: 8px; font-size: 11px;">{act_mode}</td>
                <td style="padding: 8px; font-size: 11px;">{act_status}</td>
            </tr>
            """
    else:
        fw_rows_html = "<tr><td colspan='5' style='text-align:center; padding:15px; color:#64748b;'>No firewall actions executed in current session.</td></tr>"

    blocked_list_str = ", ".join(sorted(list(set(sim_blocked + real_blocked)))) if (sim_blocked or real_blocked) else "None"

    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CYBERGRAPH COMMAND CENTER — Incident Security Audit Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #1e293b; line-height: 1.5; background: #f8fafc; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
            h1 {{ color: #0f172a; border-bottom: 3px solid #0284c7; padding-bottom: 12px; margin-top: 0; font-size: 26px; }}
            h2 {{ color: #1e293b; font-size: 18px; margin-top: 24px; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; }}
            .meta-bar {{ display: flex; justify-content: space-between; font-size: 13px; color: #64748b; margin-bottom: 20px; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }}
            .kpi-box {{ background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; text-align: center; }}
            .kpi-val {{ font-size: 24px; font-weight: 800; color: #0f172a; margin-top: 4px; }}
            .kpi-lbl {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
            th {{ background: #f8fafc; padding: 10px; text-align: left; font-weight: 700; color: #475569; border-bottom: 2px solid #cbd5e1; }}
            td {{ vertical-align: top; }}
            .status-box {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 14px; margin-bottom: 20px; font-size: 13px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CYBERGRAPH COMMAND CENTER — Incident Security Audit Report</h1>
            <div class="meta-bar">
                <span><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span><strong>Active Source:</strong> {st['current_data_source']}</span>
                <span><strong>Auditor Mode:</strong> Real-Time Automated Defense Inspection</span>
            </div>

            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-lbl">Total Evaluated Flows</div>
                    <div class="kpi-val">{st['total_flows']}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-lbl">Level 1 (Normal)</div>
                    <div class="kpi-val" style="color: #10b981;">{st['level1_normal']}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-lbl">Level 2 (Suspicious)</div>
                    <div class="kpi-val" style="color: #f59e0b;">{st['level2_suspicious']}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-lbl">Level 3 (Critical)</div>
                    <div class="kpi-val" style="color: #ef4444;">{st['level3_critical']}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-lbl">Peak Risk Observed</div>
                    <div class="kpi-val" style="color: #6366f1;">{st.get('peak_risk', 0.0):.4f}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-lbl">Firewall Rules Applied</div>
                    <div class="kpi-val" style="color: #0284c7;">{st['real_os_rules_applied']}</div>
                </div>
            </div>

            <div class="status-box">
                <h3 style="margin-top:0; color:#1e40af; font-size:15px;">Active Firewall & Containment Status</h3>
                <div><strong>Enforcement Mode:</strong> {'REAL OS FIREWALL' if is_real_enabled else 'SIMULATED ACL / RECOMMENDED APPROVAL'}</div>
                <div><strong>Currently Blocked IP Addresses:</strong> {blocked_list_str}</div>
                <div><strong>Real Windows Firewall Rules Active:</strong> {len(real_blocked)} | <strong>Simulated In-Memory Blocks:</strong> {len(sim_blocked)}</div>
            </div>

            <h2>1. Detected Threat Events & Diagnostic Findings</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 140px;">Timestamp</th>
                        <th style="width: 120px;">Severity & Level</th>
                        <th style="width: 180px;">Source & Destination</th>
                        <th>Evidence & Model Findings</th>
                        <th style="width: 220px;">Recommended Action</th>
                        <th style="width: 120px;">Firewall Status</th>
                    </tr>
                </thead>
                <tbody>
                    {threat_rows_html}
                </tbody>
            </table>

            <h2>2. Firewall Action & Response History</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 140px;">Timestamp</th>
                        <th style="width: 90px;">Action</th>
                        <th style="width: 160px;">Target IP</th>
                        <th>Execution Mode</th>
                        <th>Status Result</th>
                    </tr>
                </thead>
                <tbody>
                    {fw_rows_html}
                </tbody>
            </table>

            <h2>3. MITRE ATT&CK Contextual Framework Reference</h2>
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px; font-size:13px;">
                <p><strong>Separation of Concerns:</strong> XGBoost, Autoencoder, Fusion Layer, and Dynamic Temporal GNN compute intrinsic anomaly and graph edge risks. The MITRE ATT&CK engine deterministically maps corroborated evidence to adversary tactics and techniques:</p>
                <ul style="margin:6px 0 0 20px;">
                    <li><strong>T1046 — Network Service Discovery (Discovery):</strong> Identified via BARI route intelligence and multi-port probing patterns.</li>
                    <li><strong>T1110 — Brute Force (Credential Access):</strong> High-frequency connection attempts to authentication services (SSH, RDP, SMB).</li>
                    <li><strong>T1498 — Network Denial of Service (Impact):</strong> Volumetric throughput and rapid packet rate anomalies.</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return Response(report_html, mimetype="text/html")


@app.route("/api/execute_real_block", methods=["POST"])
def execute_real_block():
    init_engines()
    if not is_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized execute_real_block attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    target_ip = request.args.get("ip") or request.args.get("target_ip")
    if not target_ip and request.is_json:
        data = request.get_json(silent=True) or {}
        target_ip = data.get("ip") or data.get("target_ip")

    if not target_ip:
        return jsonify({"status": "ERROR", "message": "Missing IP parameter."}), 400

    clean_ip = sanitize_target_ip(target_ip)
    if not clean_ip:
        return jsonify({"status": "ERROR", "message": f"Invalid IP address: '{target_ip}'"}), 400

    res = fw_executor.execute_block(clean_ip, force_real=True)
    if res.get("status") in ["REAL_FIREWALL_RULE_APPLIED", "REAL_FIREWALL_RULE_ELEVATED"] or res.get("execution_mode") in ["REAL_OS_FIREWALL", "REAL_OS_FIREWALL_ELEVATED"]:
        with state_lock:
            shared_state["stats"]["real_os_rules_applied"] += 1

    return jsonify(res)


@app.route("/api/unblock_ip", methods=["POST"])
def api_unblock_ip():
    init_engines()
    if not is_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized unblock_ip attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    target_ip = request.args.get("ip") or request.args.get("target_ip")
    if not target_ip and request.is_json:
        data = request.get_json(silent=True) or {}
        target_ip = data.get("ip") or data.get("target_ip")

    if not target_ip:
        return jsonify({"status": "ERROR", "message": "Missing IP parameter."}), 400

    clean_ip = sanitize_target_ip(target_ip)
    if not clean_ip:
        return jsonify({"status": "ERROR", "message": f"Invalid IP address: '{target_ip}'"}), 400

    res = fw_executor.execute_unblock(clean_ip)
    return jsonify(res)


@app.route("/api/verify_blocked", methods=["GET"])
def api_verify_blocked():
    init_engines()
    target_ip = request.args.get("ip") or request.args.get("target_ip")
    if not target_ip:
        return jsonify({"status": "ERROR", "message": "Missing IP parameter."}), 400

    clean_ip = sanitize_target_ip(target_ip)
    if not clean_ip:
        return jsonify({"status": "ERROR", "message": f"Invalid IP address: '{target_ip}'"}), 400

    return jsonify(fw_executor.verify_blocked(clean_ip))


@app.before_request
def handle_device_access_request():
    if request.path.startswith("/static") or request.path.startswith("/favicon"):
        return

    client_ip = request.remote_addr or "127.0.0.1"
    # Never trust X-Forwarded-For unless explicitly configured via CYBERGRAPH_TRUST_PROXY=1
    if os.environ.get("CYBERGRAPH_TRUST_PROXY") == "1" and request.headers.get("X-Forwarded-For"):
        fwd = request.headers.get("X-Forwarded-For").split(",")[0].strip()
        if fwd:
            client_ip = fwd

    raw_token = request.cookies.get("cybergraph_device_token")
    if not raw_token or not verify_signed_device_token(raw_token, client_ip):
        token = generate_signed_device_token(client_ip)
        g.new_device_token = token
    else:
        token = raw_token
        g.new_device_token = None

    g.client_device_token = token
    g.client_ip = client_ip

    process_client_request(client_ip=client_ip, device_token=token)


@app.after_request
def set_security_headers_and_cookie(response):
    if hasattr(g, "new_device_token") and g.new_device_token:
        is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"
        response.set_cookie(
            "cybergraph_device_token",
            g.new_device_token,
            max_age=30 * 86400,
            httponly=True,
            samesite="Strict",
            secure=is_secure,
        )

    # Security Headers to prevent framing, MIME sniffing, and unauthorized script injection
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com https://fonts.gstatic.com data:; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def is_admin_request() -> bool:
    """Determine if current HTTP request originates from authenticated CyberGraph Admin.
    Validates token via:
      1. X-Admin-Token or X-CSRF-Token header
      2. Authorization: Bearer <token> header
      3. cybergraph_admin_token cookie
      4. JSON body or query param 'admin_token'
      5. Explicit test environment bypass via CYBERGRAPH_ALLOW_LOCALHOST_ADMIN=1.
    """
    token_candidate = (
        request.headers.get("X-Admin-Token")
        or request.headers.get("X-CSRF-Token")
        or request.cookies.get("cybergraph_admin_token")
    )
    if not token_candidate and request.headers.get("Authorization", "").startswith("Bearer "):
        token_candidate = request.headers.get("Authorization", "")[7:].strip()
    if not token_candidate and request.is_json:
        data = request.get_json(silent=True) or {}
        token_candidate = data.get("admin_token") or data.get("token")
    if not token_candidate:
        token_candidate = request.args.get("admin_token") or request.args.get("token")

    if token_candidate and hmac.compare_digest(str(token_candidate).strip(), ADMIN_TOKEN):
        return True

    # Test environment bypass flag
    if os.environ.get("CYBERGRAPH_ALLOW_LOCALHOST_ADMIN") == "1":
        client_ip = request.remote_addr or "127.0.0.1"
        srv_ip = get_server_local_ip()
        if client_ip in ("127.0.0.1", "::1", "localhost", "0.0.0.0") or (srv_ip and client_ip == srv_ip):
            return True

    return False


@app.route("/api/device_requests", methods=["GET"])
def api_device_requests():
    init_engines()
    return jsonify({
        "status": "SUCCESS",
        "is_admin": is_admin_request(),
        "authorized_devices": get_authorized_devices(),
        "pending_requests": get_pending_requests(),
    })


@app.route("/api/approve_device", methods=["POST"])
def api_approve_device():
    init_engines()
    if not is_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized approve_device attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    req_id = request.args.get("request_id") or request.args.get("device_id") or request.args.get("ip")
    if not req_id and request.is_json:
        data = request.get_json(silent=True) or {}
        req_id = data.get("request_id") or data.get("device_id") or data.get("ip")

    if not req_id:
        return jsonify({"status": "ERROR", "message": "Missing request_id, device_id, or ip parameter"}), 400

    res = approve_device_request(req_id)
    if res:
        with state_lock:
            shared_state["state_version"] += 1
        return jsonify({"status": "SUCCESS", "approved_device": res})
    return jsonify({"status": "ERROR", "message": f"Pending request or device '{req_id}' not found"}), 404


@app.route("/api/reject_device", methods=["POST"])
def api_reject_device():
    init_engines()
    if not is_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized reject_device attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    req_id = request.args.get("request_id") or request.args.get("device_id") or request.args.get("ip")
    if not req_id and request.is_json:
        data = request.get_json(silent=True) or {}
        req_id = data.get("request_id") or data.get("device_id") or data.get("ip")

    if not req_id:
        return jsonify({"status": "ERROR", "message": "Missing request_id, device_id, or ip parameter"}), 400

    ok = reject_device_request(req_id)
    if ok:
        with state_lock:
            shared_state["state_version"] += 1
        return jsonify({"status": "SUCCESS", "message": f"Device request '{req_id}' rejected"})
    return jsonify({"status": "ERROR", "message": f"Request '{req_id}' not found"}), 404


@app.route("/api/revoke_device", methods=["POST"])
def api_revoke_device():
    init_engines()
    if not is_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized revoke_device attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    req_id = request.args.get("device_id") or request.args.get("ip") or request.args.get("request_id")
    if not req_id and request.is_json:
        data = request.get_json(silent=True) or {}
        req_id = data.get("device_id") or data.get("ip") or data.get("request_id")

    if not req_id:
        return jsonify({"status": "ERROR", "message": "Missing device_id or ip parameter"}), 400

    ok = revoke_device_authorization(req_id)
    if ok:
        with state_lock:
            shared_state["state_version"] += 1
        return jsonify({"status": "SUCCESS", "message": f"Device '{req_id}' authorization revoked"})
    return jsonify({"status": "ERROR", "message": f"Device '{req_id}' not found or cannot be revoked"}), 404


if __name__ == "__main__":
    init_engines()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    host = os.environ.get("CYBERGRAPH_HOST", "127.0.0.1")
    port = int(os.environ.get("CYBERGRAPH_PORT", "5000"))
    print("\n" + "=" * 72)
    print("CYBERGRAPH COMMAND CENTER — INITIALIZED")
    print(f"Server Binding : http://{host}:{port}")
    print(f"[SECURITY] ADMIN TOKEN : {ADMIN_TOKEN}")
    print("Phase 2 MITRE ATT&CK integration: ACTIVE")
    print("Supported techniques: T1046, T1110, T1498")
    print("=" * 72 + "\n")
    app.run(host=host, port=port, debug=False)