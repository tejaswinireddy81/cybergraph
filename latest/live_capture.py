"""
live_capture.py
===============
AI-Powered Network Traffic Analyzer — LIVE PACKET CAPTURE

Runs Scapy packet sniffing directly on the active Wi-Fi / Ethernet interface.
Converts captured packets into 49 NetFlow features via FlowExtractor, validates schema,
and appends live flows to live_flows for real-time model inference.
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

from flow_extractor import FlowExtractor
from modules.live_traffic_engine import LiveTrafficEngine

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_FLOWS        = 10_000   # rolling window kept in memory
FLUSH_INTERVAL_S = 2.0      # flush expired flows every N seconds
LOG_INTERVAL_S   = 10.0     # print pipeline diagnostics every N seconds

live_flows = deque(maxlen=MAX_FLOWS)

capture_status = {
    "running"           : False,
    "iface"             : None,
    "packets"           : 0,
    "ipv4_packets"      : 0,
    "active_flows"      : 0,
    "flows_finalized"   : 0,
    "flows"             : 0,
    "errors"            : 0,
    "last_error"        : "",
}

_extractor: Optional[FlowExtractor] = None
_engine: Optional[LiveTrafficEngine] = None

_flows_lock  = threading.Lock()
_status_lock = threading.Lock()
_stop_event  = threading.Event()


def _get_scapy_ifaces_list():
    try:
        from scapy.interfaces import IFACES
        result = []
        for i, (guid, intf) in enumerate(IFACES.items()):
            desc = getattr(intf, "description", "") or getattr(intf, "network_name", "") or str(guid)
            result.append((i, intf, desc))
        if result:
            return result
    except Exception:
        pass
    return []


def list_interfaces() -> list:
    raw = _get_scapy_ifaces_list()
    result = []
    for i, iface_obj, desc in raw:
        name_str = iface_obj if isinstance(iface_obj, str) else str(iface_obj)
        result.append((i, name_str, desc))
    return result


def _resolve_scapy_iface(selected: Optional[str]):
    if selected is None:
        return None
    raw = _get_scapy_ifaces_list()
    for _i, iface_obj, desc in raw:
        if isinstance(iface_obj, str):
            if selected in (iface_obj, desc):
                return iface_obj
        else:
            intf_name = getattr(iface_obj, "name", "")
            intf_desc = getattr(iface_obj, "description", "")
            intf_network = getattr(iface_obj, "network_name", "")
            candidates = {intf_name, intf_desc, intf_network, str(iface_obj), desc}
            if selected in candidates:
                return iface_obj
    return selected


def _process_flow(feature_dict: dict):
    global _engine
    if _engine is None:
        _engine = LiveTrafficEngine()

    feature_dict["_capture_time"] = time.time()
    with _flows_lock:
        live_flows.append(feature_dict)

    with _status_lock:
        capture_status["flows"] += 1


def _run_flush(extractor: FlowExtractor):
    while not _stop_event.is_set():
        time.sleep(FLUSH_INTERVAL_S)
        if _stop_event.is_set():
            break

        try:
            expired = extractor.flush_expired()
        except Exception as exc:
            expired = []

        n = len(expired)
        if n:
            with _status_lock:
                capture_status["flows_finalized"] += n
                capture_status["active_flows"] = extractor.active_flow_count()
            for flow_feat in expired:
                _process_flow(flow_feat)


def _run_capture(scapy_iface, extractor: FlowExtractor):
    from scapy.sendrecv import sniff

    with _status_lock:
        capture_status["running"] = True
        iface_label = str(scapy_iface) if scapy_iface else "default"
        capture_status["iface"] = iface_label

    def _handle_packet(pkt):
        with _status_lock:
            capture_status["packets"] += 1
        try:
            result = extractor.add_packet(pkt)
        except Exception:
            return
        if result is not None:
            _process_flow(result)

    def _stop_filter(_):
        return _stop_event.is_set()

    sniff_kwargs = dict(prn=_handle_packet, store=False, stop_filter=_stop_filter)
    if scapy_iface is not None:
        sniff_kwargs["iface"] = scapy_iface

    try:
        sniff(**sniff_kwargs)
    except Exception as exc:
        with _status_lock:
            capture_status["errors"] += 1
            capture_status["last_error"] = str(exc)
    finally:
        with _status_lock:
            capture_status["running"] = False


_capture_thread: Optional[threading.Thread] = None
_flush_thread:   Optional[threading.Thread] = None


def start_capture(iface: Optional[str] = None):
    global _capture_thread, _flush_thread, _extractor, _engine

    _stop_event.clear()
    _extractor = FlowExtractor()
    _engine = LiveTrafficEngine()

    scapy_iface = _resolve_scapy_iface(iface)

    _flush_thread = threading.Thread(target=_run_flush, args=(_extractor,), daemon=True, name="FlowFlushThread")
    _flush_thread.start()

    _capture_thread = threading.Thread(target=_run_capture, args=(scapy_iface, _extractor), daemon=True, name="LiveCaptureThread")
    _capture_thread.start()
    logger.info("Live capture thread launched.")


def stop_capture():
    _stop_event.set()
    for t in (_capture_thread, _flush_thread):
        if t and t.is_alive():
            t.join(timeout=10)
