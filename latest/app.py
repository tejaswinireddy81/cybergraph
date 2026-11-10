"""
app.py
======
CYBERGRAPH COMMAND CENTER — Explainable AI Network Threat Detection & Response
Complete Implementation with Server-Authoritative Pipeline & Real-Time Synchronization
"""

import sys
sys.modules["app"] = sys.modules[__name__]

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
from flask import Flask, render_template, render_template_string, jsonify, request, Response, g, make_response
from config.secrets_manager import get_hmac_secret, get_admin_token, get_session_secret
from modules.auth_manager import (
    create_admin_session,
    terminate_session,
    validate_session,
    verify_admin_request,
    verify_csrf_protection,
    verify_read_viewer_access,
    generate_signed_device_token as am_generate_signed_device_token,
    verify_signed_device_token as am_verify_signed_device_token,
    generate_csrf_token_for_session,
)
from modules.trusted_model_loader import safe_load_joblib, safe_load_torch
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

# Cryptographic Persistent Secrets & Admin Token
ADMIN_TOKEN = get_admin_token()
HMAC_SECRET = get_hmac_secret()

def generate_signed_device_token(client_ip: str) -> str:
    return am_generate_signed_device_token(client_ip)

def verify_signed_device_token(token: str, client_ip: str) -> bool:
    return am_verify_signed_device_token(token, client_ip)

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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max payload limit

# --- Global API Error Handlers ---
@app.errorhandler(400)
def handle_bad_request(e):
    if request.path.startswith("/api/"):
        return jsonify({"status": "ERROR", "message": f"Bad Request: {str(e)}"}), 400
    return e, 400

@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"status": "ERROR", "message": f"Endpoint Not Found: {request.path}"}), 404
    return e, 404

@app.errorhandler(405)
def handle_method_not_allowed(e):
    if request.path.startswith("/api/"):
        return jsonify({"status": "ERROR", "message": f"Method Not Allowed: {request.method} on {request.path}"}), 405
    return e, 405

@app.errorhandler(413)
def handle_payload_too_large(e):
    if request.path.startswith("/api/"):
        return jsonify({"status": "ERROR", "message": "Payload Too Large (Maximum body size is 16MB)"}), 413
    return e, 413

@app.errorhandler(500)
def handle_internal_server_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"status": "ERROR", "message": "Internal Server Error"}), 500
    return e, 500

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
            scalers = safe_load_joblib(SCALERS_PATH)
            gnn_model = DynamicBipartiteTemporalGNN(
                host_dim=8, service_dim=8, edge_dim=9, hidden_dim=32
            )
            gnn_model.load_state_dict(safe_load_torch(GNN_MODEL_PATH, weights_only=True))
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


TEMPLATES_DIR = os.path.join(HERE, "templates")

def _load_html_template() -> str:
    path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>CyberGraph Template Not Found</h1>"

HTML_TEMPLATE = _load_html_template()


def stream_events():
    return _stream_events_impl()


def _stream_events_impl():
    """Server-Sent Events (SSE) stream providing real-time push updates (Viewer Access Protected).
    Yields full state snapshot events whenever state_version increments,
    with periodic heartbeats to maintain persistent connection.
    """
    init_engines()
    if not verify_read_viewer_access(lambda tok: tok in _authorized_monitored_devices):
        return Response("event: error\ndata: {\"error\": \"Unauthorized viewer\"}\n\n", mimetype="text/event-stream", status=401)

    def event_stream():
        last_version = -1
        last_heartbeat = time.time()
        while True:
            current_time = time.time()
            with state_lock:
                current_version = shared_state.get("state_version", 0)
                if current_version != last_version:
                    snapshot = copy.deepcopy(shared_state)
                    last_version = current_version
                    payload = json.dumps(snapshot)
                    yield f"event: update\ndata: {payload}\n\n"
                    last_heartbeat = current_time

            # Keepalive ping every 5 seconds
            if current_time - last_heartbeat >= 5.0:
                yield "event: ping\ndata: {}\n\n"
                last_heartbeat = current_time

            time.sleep(0.4)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


def is_physical_network_active() -> bool:
    """Return True if an active physical network interface (Wi-Fi / Ethernet) is UP with a valid private IPv4."""
    try:
        import psutil
        import ipaddress
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        virtual_kw = ("virtual", "vethernet", "loopback", "wsl", "host-only", "vmware", "virtualbox", "wan miniport", "bluetooth", "teredo", "isatap")
        for name, stat in stats.items():
            if not stat.isup:
                continue
            if any(kw in name.lower() for kw in virtual_kw):
                continue
            for snic in addrs.get(name, []):
                if getattr(snic, "family", None) == 2:
                    try:
                        ip_obj = ipaddress.IPv4Address(snic.address)
                        if ip_obj.is_private and not ip_obj.is_loopback and not ip_obj.is_link_local:
                            return True
                    except Exception:
                        pass
    except Exception:
        pass
    return False


def is_live_capture_available() -> bool:
    """Return True if Scapy/Npcap live capture is initialized and physical network is connected."""
    if not is_physical_network_active():
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
            if not dev.get("is_server"):
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
    # Automatic capture watchdog: if physical network is active, ensure sniffer thread is running
    if is_physical_network_active():
        cap = get_capture_status()
        if not cap.get("running"):
            try:
                start_capture(iface=iface)
            except Exception as _c_exc:
                logger.debug("Auto-restart capture notice: %s", _c_exc)


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
    recent_live = get_recent_live_flows(window_seconds=1.5)
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
        wifi_connected = is_physical_network_active()

        if wifi_connected:
            data_source = "LIVE WI-FI PACKETS (PHYSICAL CAPTURE ACTIVE)"
            buf_count = len(recent_live)
            if has_recent_live:
                batch_records = recent_live[-batch_sz:]
            else:
                # Wi-Fi is physically connected; idle baseline: monitor server & gateway communication
                pool = get_stream_pool()
                if len(pool) > 0:
                    benign_sub = pool[pool["label_enc"] == 0] if "label_enc" in pool.columns else pool
                    batch_df = benign_sub.sample(n=min(batch_sz, len(benign_sub))).copy()
                    srv_ip = get_server_local_ip() or "192.168.1.37"
                    batch_df["IPV4_SRC_ADDR"] = srv_ip
                    batch_records = batch_df.to_dict(orient="records")
                else:
                    batch_records = []
        else:
            # Physical Wi-Fi is disconnected: automatically stream offline dataset!
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
    # Zero credential exposure: Never embed ADMIN_TOKEN in HTML/JS responses
    return render_template("index.html", is_admin=is_admin)


# --- Dedicated Authentication & CSRF Endpoints ---

# --- Register Modular Flask Blueprints ---
from routes.auth_routes import auth_bp
from routes.admin_routes import admin_bp
from routes.device_routes import device_bp
from routes.stream_routes import stream_bp
from routes.report_routes import report_bp

app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(device_bp)
app.register_blueprint(stream_bp)
app.register_blueprint(report_bp)

# Top-level backward-compatible function aliases for test modules
from routes.admin_routes import execute_real_block, api_unblock_ip, api_verify_blocked, api_blocks_history
from routes.device_routes import approve_device, reject_device, revoke_device, api_device_requests
from routes.stream_routes import stream_step, stream_events, set_mode
from routes.report_routes import export_report, attack_journey
from routes.auth_routes import api_auth_login, api_auth_logout, api_auth_status, api_auth_csrf


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
        "default-src 'self' https://cdn.jsdelivr.net https://fonts.googleapis.com https://fonts.gstatic.com data:; "
        "script-src 'self' https://cdn.jsdelivr.net; "
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
    Enforces strict security:
      1. Validates authenticated session cookie ('cybergraph_session') or 'X-Admin-Token' / 'Bearer <token>' headers.
      2. Strictly REJECTS credentials or tokens passed in URL query parameters.
      3. Verifies CSRF protection for state-changing requests.
    """
    return verify_admin_request()

# Backward-compatible function aliases (handled canonically by device_bp)
api_approve_device = approve_device
api_reject_device = reject_device
api_revoke_device = revoke_device


if __name__ == "__main__":
    init_engines()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    host = os.environ.get("CYBERGRAPH_HOST", "0.0.0.0")
    port = int(os.environ.get("CYBERGRAPH_PORT", "5000"))
    print("\n" + "=" * 72)
    print("CYBERGRAPH COMMAND CENTER — INITIALIZED")
    print(f"Server Binding : http://{host}:{port}")
    print(f"[SECURITY] ADMIN TOKEN : {ADMIN_TOKEN}")
    print("Phase 2 MITRE ATT&CK integration: ACTIVE")
    print("Supported techniques: T1046, T1110, T1498")
    print("=" * 72 + "\n")
    app.run(host=host, port=port, debug=False)