"""
modules/arp_monitor.py
======================
OPTIONAL, STANDALONE ARP anomaly detector.

ARP is not an IPv4 flow, so it is deliberately OUTSIDE the flow-based ML model (FlowExtractor drops it before
the model sees anything). This module is the separate place for ARP-level checks.

Isolation guarantees (so it cannot interfere with the current pipeline)
  * It imports nothing from the app, the flow extractor, the models or the firewall code.
  * Nothing imports it; it is not started by app.py. It only runs when you call it.
  * It never blocks, never touches the firewall and never writes files; `observe()` returns alert objects.
  * Disabled by default: `is_enabled()` reads CYBERGRAPH_ARP_MONITOR (default "0") for whoever wires it in later.

Detections (trust-on-first-use baseline, optional static bindings)
  IP_MAC_CHANGE         an IP that was bound to one MAC is now claimed by another   (classic ARP spoofing)
  GATEWAY_MAC_CHANGE    same, for a configured gateway IP                           (critical)
  MAC_MANY_IPS          one MAC claims many different IPs within a window           (poisoning many hosts)
  GRATUITOUS_FLOOD      many gratuitous ARPs from one MAC within a window
  ARP_SWEEP             one sender asks for many different target IPs in a window   (ARP scan)
  UNSOLICITED_REPLY     a reply nobody asked for (low severity; many stacks do this legitimately)
  INVALID_SENDER        broadcast/multicast sender MAC, or reply from 0.0.0.0

False-positive notes: DHCP/DAD probes (request from 0.0.0.0) are ignored; routers doing proxy-ARP legitimately
claim many IPs, so allow-list their MAC with `trusted_macs`; VRRP/failover legitimately moves an IP between MACs.
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Set

BROADCAST_MAC = "ff:ff:ff:ff:ff:ff"


def is_enabled() -> bool:
    """Whoever integrates this module should honour the flag. Default: off."""
    return os.environ.get("CYBERGRAPH_ARP_MONITOR", "0") == "1"


def _norm_mac(mac: str) -> str:
    return (mac or "").strip().lower().replace("-", ":")


@dataclass(frozen=True)
class ArpEvent:
    ts: float
    op: str            # "request" or "reply"
    sender_ip: str
    sender_mac: str
    target_ip: str
    target_mac: str = "00:00:00:00:00:00"


@dataclass
class ArpAlert:
    kind: str
    severity: str      # low | medium | high | critical
    ts: float
    ip: Optional[str]
    mac: Optional[str]
    details: Dict[str, object] = field(default_factory=dict)


class ArpMonitor:
    def __init__(self, gateway_ips: Optional[Set[str]] = None, trusted_macs: Optional[Set[str]] = None,
                 static_bindings: Optional[Dict[str, str]] = None, window_s: float = 10.0,
                 max_ips_per_mac: int = 5, max_gratuitous: int = 5, max_sweep_targets: int = 20,
                 reply_match_s: float = 5.0, max_table: int = 10_000,
                 on_alert: Optional[Callable[[ArpAlert], None]] = None):
        self.gateways = set(gateway_ips or ())
        self.trusted_macs = {_norm_mac(m) for m in (trusted_macs or ())}
        self.bindings: Dict[str, str] = {ip: _norm_mac(m) for ip, m in (static_bindings or {}).items()}
        self.window_s, self.max_ips_per_mac = window_s, max_ips_per_mac
        self.max_gratuitous, self.max_sweep_targets = max_gratuitous, max_sweep_targets
        self.reply_match_s, self.max_table = reply_match_s, max_table
        self.on_alert = on_alert
        self._lock = threading.Lock()
        self._mac_ips: Dict[str, Deque] = defaultdict(deque)      # mac -> deque[(ts, ip)]
        self._gratuitous: Dict[str, Deque] = defaultdict(deque)   # mac -> deque[ts]
        self._requests: Dict[str, Deque] = defaultdict(deque)     # sender_mac -> deque[(ts, target_ip)]
        self._asked: Dict[tuple, float] = {}                      # (asker_ip, asked_ip) -> ts of last request
        self._last_alert: Dict[tuple, float] = {}                 # de-duplication

    # -------------------------------------------------------------------------------------------- helpers
    def _prune(self, dq: Deque, now: float, key=lambda x: x):
        while dq and now - key(dq[0]) > self.window_s:
            dq.popleft()

    def _emit(self, out: List[ArpAlert], alert: ArpAlert, dedup_key: tuple):
        last = self._last_alert.get(dedup_key)
        if last is not None and alert.ts - last < self.window_s:
            return
        self._last_alert[dedup_key] = alert.ts
        out.append(alert)
        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception:  # an alert handler must never break monitoring
                pass

    # ------------------------------------------------------------------------------------------ main entry
    def observe(self, ev: ArpEvent) -> List[ArpAlert]:
        """Feed one ARP packet; returns the alerts it triggered (usually none)."""
        out: List[ArpAlert] = []
        mac, ip, now = _norm_mac(ev.sender_mac), ev.sender_ip, ev.ts
        with self._lock:
            if len(self.bindings) > self.max_table:   # bounded memory: drop learned (not static) state
                self.bindings = {k: v for k, v in self.bindings.items() if k in self.gateways}
            # sender sanity
            if mac == BROADCAST_MAC or int(mac.split(":")[0] or "0", 16) & 1:
                self._emit(out, ArpAlert("INVALID_SENDER", "high", now, ip, mac,
                                         {"reason": "broadcast/multicast sender MAC"}), ("INV", mac))
                return out
            if ev.op == "reply" and ip == "0.0.0.0":  # nosec B104 - an ARP sender address, not a socket bind
                self._emit(out, ArpAlert("INVALID_SENDER", "high", now, ip, mac, {"reason": "reply from 0.0.0.0"}),
                           ("INV0", mac))
                return out
            probe = ev.op == "request" and ip == "0.0.0.0"  # nosec B104  DHCP / duplicate-address detection: legitimate
            if probe:
                return out

            # 1) binding changes (spoofing)
            known = self.bindings.get(ip)
            if known is None:
                self.bindings[ip] = mac                       # trust on first use
            elif known != mac and mac not in self.trusted_macs:
                sev = "critical" if ip in self.gateways else "high"
                kind = "GATEWAY_MAC_CHANGE" if ip in self.gateways else "IP_MAC_CHANGE"
                self._emit(out, ArpAlert(kind, sev, now, ip, mac, {"previous_mac": known, "new_mac": mac}),
                           (kind, ip, mac))

            # 2) one MAC claiming many IPs
            dq = self._mac_ips[mac]
            dq.append((now, ip))
            self._prune(dq, now, key=lambda x: x[0])
            ips = {i for _, i in dq}
            if mac not in self.trusted_macs and len(ips) > self.max_ips_per_mac:
                self._emit(out, ArpAlert("MAC_MANY_IPS", "high", now, ip, mac,
                                         {"distinct_ips": len(ips), "window_s": self.window_s}), ("MANY", mac))

            gratuitous = ev.sender_ip == ev.target_ip
            if gratuitous:
                g = self._gratuitous[mac]
                g.append(now)
                self._prune(g, now)
                if len(g) > self.max_gratuitous and mac not in self.trusted_macs:
                    self._emit(out, ArpAlert("GRATUITOUS_FLOOD", "medium", now, ip, mac,
                                             {"count": len(g), "window_s": self.window_s}), ("GRAT", mac))

            if ev.op == "request":
                r = self._requests[mac]
                r.append((now, ev.target_ip))
                self._prune(r, now, key=lambda x: x[0])
                targets = {t for _, t in r}
                self._asked[(ip, ev.target_ip)] = now
                if len(targets) > self.max_sweep_targets and mac not in self.trusted_macs:
                    self._emit(out, ArpAlert("ARP_SWEEP", "medium", now, ip, mac,
                                             {"distinct_targets": len(targets), "window_s": self.window_s}), ("SWEEP", mac))
            elif ev.op == "reply" and not gratuitous:
                asked_at = self._asked.get((ev.target_ip, ip))
                if asked_at is None or now - asked_at > self.reply_match_s:
                    self._emit(out, ArpAlert("UNSOLICITED_REPLY", "low", now, ip, mac,
                                             {"target_ip": ev.target_ip}), ("UNSOL", ip, mac))
        return out

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            return dict(self.bindings)


# ----------------------------------------------------------------------- optional scapy glue (lazy, not auto-run)
def event_from_scapy(pkt) -> Optional[ArpEvent]:
    """Convert a scapy ARP packet to an ArpEvent (None if the packet is not ARP). Imports scapy lazily."""
    from scapy.layers.l2 import ARP
    if not pkt.haslayer(ARP):
        return None
    a = pkt[ARP]
    return ArpEvent(ts=float(getattr(pkt, "time", 0.0)), op="request" if a.op == 1 else "reply",
                    sender_ip=a.psrc, sender_mac=a.hwsrc, target_ip=a.pdst, target_mac=a.hwdst)


def sniff_arp(monitor: ArpMonitor, iface: Optional[str] = None, timeout: Optional[float] = None) -> None:
    """Blocking helper for an explicit, separate ARP-monitor process. Never called by the application."""
    from scapy.sendrecv import sniff

    def _cb(pkt):
        ev = event_from_scapy(pkt)
        if ev is not None:
            monitor.observe(ev)

    sniff(filter="arp", prn=_cb, store=False, iface=iface, timeout=timeout)
