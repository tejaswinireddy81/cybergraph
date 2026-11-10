"""
evaluation/synthetic_traffic.py
===============================
Deterministic SYNTHETIC traffic through the real FlowExtractor, used until lab captures are approved.

IMPORTANT LIMITS (state them whenever results from this module are shown)
  * It is a model of traffic, not traffic. Packet sizes/timings are drawn from simple distributions chosen
    from protocol knowledge (handshake shapes, typical payload sizes); they will not match every real stack.
  * It is good for: exercising the pipeline, comparing feature sets / the TCP floor on controlled shapes,
    measuring how a model treats protocols it never saw (QUIC, IGMP, NTP ...), and unit tests.
  * It is NOT evidence of real-world false-positive rate. That needs the approved lab captures.

No packet is ever sent on a network: packets are built in memory and fed to the extractor.

Timing: FlowExtractor stamps packets with wall-clock time.time(), so a controllable clock is injected while
the packets are fed; durations / IATs / throughputs are therefore reproducible for a given seed.

Every packet also carries its CAPTURE timestamp in `pkt.time` (as a real sniff does). Optionally a processing-delay
model can be applied (`proc_cost_s`, `proc_jitter_s`): the extractor's wall clock then lags/bunches relative to the
capture time, which lets the capture-vs-processing timestamp difference be measured. With both at 0 (default) the two
clocks are identical.

TCP realism (revision 3): per-direction sequence numbers advance with payload (SYN/FIN consume one), pure ACKs repeat the
sequence number as real stacks do, and ACK/data options depend on the OS profile (a Windows client sends bare 40-byte
ACKs; Linux/macOS clients send timestamp options, 52 bytes; servers are modelled as timestamp-enabled).
"""
from __future__ import annotations

import time as _real_time
import zlib
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import flow_extractor as _fx
from flow_extractor import FlowExtractor
from train_xgboost_ae import FEATURE_COLS

from scapy.all import ICMP, IP, TCP, UDP, Ether, Raw  # noqa: E402
from scapy.contrib.igmp import IGMP  # noqa: E402

BENIGN_CLASSES = ["dns_udp", "dns_tcp", "https_tcp", "http_tcp", "other_tcp", "quic_udp",
                  "ntp_udp", "dhcp_udp", "mdns_udp", "ssdp_udp", "icmp_ping", "igmp"]
ATTACK_CLASSES = ["syn_scan_nmap", "syn_scan_bare", "port_sweep", "syn_flood", "udp_flood",
                  "icmp_sweep", "brute_force_like"]
ATTACK_FAMILY = {  # family names used in per-family reporting
    "syn_scan_nmap": "Scan", "syn_scan_bare": "Scan", "port_sweep": "Scan", "icmp_sweep": "Scan",
    "syn_flood": "DoS", "udp_flood": "DoS", "brute_force_like": "BruteForce",
}
OS_PROFILES = {  # TCP SYN option layouts: Windows (52 B SYN), Linux (60 B), macOS (64 B)
    "win": [("MSS", 1460), ("NOP", None), ("WScale", 8), ("NOP", None), ("NOP", None), ("SAckOK", b"")],
    "linux": [("MSS", 1460), ("SAckOK", b""), ("Timestamp", (1, 0)), ("NOP", None), ("WScale", 7)],
    "mac": [("MSS", 1460), ("NOP", None), ("WScale", 6), ("NOP", None), ("NOP", None),
            ("Timestamp", (1, 0)), ("SAckOK", b""), ("EOL", None)],
}


class _Clock:
    """Stand-in for the `time` module inside flow_extractor; only .time() is virtual."""
    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now

    def __getattr__(self, item):
        return getattr(_real_time, item)


class _Session:
    """One simulated capture session: one extractor, one clock, one client."""

    def __init__(self, rng: np.random.Generator, session_id: str, floor: int, os_name: str, t0: float,
                 extractor_factory=None, proc_cost_s: float = 0.0, proc_jitter_s: float = 0.0):
        self.rng, self.session_id, self.os_name = rng, session_id, os_name
        self.clock = _Clock()
        self.clock.now = t0
        self.capture_now = t0
        self.proc_cost_s, self.proc_jitter_s = float(proc_cost_s), float(proc_jitter_s)
        self._seqs: Dict[tuple, int] = {}
        # processing-delay noise has its OWN stream so changing the delay model never changes the generated packets
        self._proc_rng = np.random.default_rng(zlib.crc32(session_id.encode()) ^ 0x5EED)
        self.ex = extractor_factory(floor) if extractor_factory else FlowExtractor(min_pkt_floor_tcp=floor)
        self.client = f"192.168.{int(rng.integers(0, 4))}.{int(rng.integers(20, 200))}"
        self.records: List[dict] = []
        self._pending: List[dict] = []

    # ---- helpers ---------------------------------------------------------------------------
    def server_ip(self):
        r = self.rng
        return f"{int(r.choice([8, 34, 52, 93, 104, 142, 151, 172, 185]))}.{int(r.integers(1, 255))}.{int(r.integers(1, 255))}.{int(r.integers(1, 255))}"

    def feed(self, pkt, dt: float):
        self.capture_now += max(dt, 1e-6)
        if self.proc_cost_s > 0 or self.proc_jitter_s > 0:   # processing lags behind capture
            lag = self.proc_cost_s + (float(self._proc_rng.exponential(self.proc_jitter_s)) if self.proc_jitter_s > 0 else 0.0)
            self.clock.now = max(self.clock.now + lag, self.capture_now)
        else:
            self.clock.now = self.capture_now
        frame = Ether() / pkt
        frame.time = self.capture_now                       # what libpcap would stamp on the frame
        rec = self.ex.add_packet(frame)
        if rec is not None:  # the extractor exports a TCP flow at the first FIN/RST and returns it here
            self._pending.append(rec)

    def pause(self, seconds: float):
        self.capture_now += seconds
        self.clock.now = max(self.clock.now, self.capture_now)

    def finish(self, label: int, traffic_class: str, family: str):
        exported = list(self._pending)
        self._pending.clear()
        drain = getattr(self.ex, "drain_all", None)
        if drain is not None:                      # proposed extractor: finalise everything it still holds
            exported.extend(drain())
        else:
            for flow in list(self.ex._flows.values()):
                rec = self.ex._export(flow)
                if rec is not None:
                    exported.append(rec)
        for rec in exported:
            if rec is not None:
                rec = dict(rec)
                rec.update({"_label": label, "_traffic_class": traffic_class, "_family": family,
                            "_session": self.session_id, "_synthetic": True})
                self.records.append(rec)
        self.ex._flows.clear()

    def rtt(self):
        return float(self.rng.lognormal(mean=np.log(0.03), sigma=0.5))

    # ---- TCP building blocks -----------------------------------------------------------------
    def _next_seq(self, src, sport, dst, dport, advance: int = 0) -> int:
        k = (src, sport, dst, dport)
        cur = self._seqs.get(k)
        if cur is None:
            cur = int(self.rng.integers(1, 2**31))
        self._seqs[k] = (cur + advance) & 0xFFFFFFFF
        return cur

    def _opts(self, src):
        """Options on established-connection packets: Windows client -> none (bare 40-byte ACK); others -> timestamps."""
        if src == self.client and self.os_name == "win":
            return []
        return [("NOP", None), ("NOP", None), ("Timestamp", (1, 0))]

    def tcp_open(self, src, dst, sport, dport):
        r = self.rtt()
        self.feed(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="S", options=OS_PROFILES[self.os_name],
                                             seq=self._next_seq(src, sport, dst, dport, 1)), 0.001)
        self.feed(IP(src=dst, dst=src) / TCP(sport=dport, dport=sport, flags="SA",
                                             seq=self._next_seq(dst, dport, src, sport, 1),
                                             options=[("MSS", 1460), ("NOP", None), ("WScale", 7), ("SAckOK", b"")]), r)
        self.feed(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="A", options=self._opts(src),
                                             seq=self._next_seq(src, sport, dst, dport)), r / 4)

    def tcp_data(self, src, dst, sport, dport, size: int, gap: float, push=True):
        seq = self._next_seq(src, sport, dst, dport, size)
        self.feed(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="PA" if push else "A",
                                             seq=seq, options=self._opts(src)) / Raw(b"x" * size), gap)

    def tcp_pure_ack(self, src, dst, sport, dport, gap: float):
        self.feed(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="A", options=self._opts(src),
                                             seq=self._next_seq(src, sport, dst, dport)), gap)

    def tcp_close(self, src, dst, sport, dport):
        """Client-initiated teardown: FIN-ACK, server FIN-ACK, final ACK (the usual 3-packet form)."""
        self.feed(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="FA", options=self._opts(src),
                                             seq=self._next_seq(src, sport, dst, dport, 1)), 0.01)
        self.feed(IP(src=dst, dst=src) / TCP(sport=dport, dport=sport, flags="FA", options=self._opts(dst),
                                             seq=self._next_seq(dst, dport, src, sport, 1)), 0.02)
        self.feed(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="A", options=self._opts(src),
                                             seq=self._next_seq(src, sport, dst, dport)), 0.01)


# ======================================================================== benign generators
def _eph(r):
    return int(r.integers(49152, 65535))


def gen_dns_udp(s: _Session):
    r, srv = s.rng, str(s.rng.choice(["8.8.8.8", "8.8.4.4", "1.1.1.1", "9.9.9.9", "192.168.1.1"]))
    sp = _eph(r)
    q = int(r.integers(12, 40))
    s.feed(IP(src=s.client, dst=srv) / UDP(sport=sp, dport=53) / Raw(b"q" * q), 0.001)
    s.feed(IP(src=srv, dst=s.client) / UDP(sport=53, dport=sp) / Raw(b"a" * int(r.integers(q + 16, q + 160))), s.rtt())
    s.finish(0, "dns_udp", "Benign")


def gen_dns_tcp(s: _Session):
    r, srv, sp = s.rng, str(s.rng.choice(["8.8.8.8", "1.1.1.1"])), 0
    sp = _eph(r)
    s.tcp_open(s.client, srv, sp, 53)
    s.tcp_data(s.client, srv, sp, 53, int(r.integers(30, 60)), 0.001)
    s.tcp_data(s.server_ip() if False else srv, s.client, 53, sp, int(r.integers(100, 500)), s.rtt())
    s.tcp_close(s.client, srv, sp, 53)
    s.finish(0, "dns_tcp", "Benign")


def _tls_exchange(s: _Session, dport: int, rounds: int):
    r, srv, sp = s.rng, s.server_ip(), 0
    sp = _eph(r)
    s.tcp_open(s.client, srv, sp, dport)
    s.tcp_data(s.client, srv, sp, dport, int(r.integers(480, 620)), 0.001)        # ClientHello
    for _ in range(int(r.integers(3, 5))):                                        # ServerHello + certificates
        s.tcp_data(srv, s.client, dport, sp, 1448, s.rtt() / 3, push=False)
    s.tcp_data(s.client, srv, sp, dport, int(r.integers(60, 130)), 0.002)          # Finished
    for _ in range(rounds):
        s.tcp_data(s.client, srv, sp, dport, int(r.integers(40, 700)), float(r.exponential(0.2)))
        for _ in range(int(r.integers(1, 6))):
            s.tcp_data(srv, s.client, dport, sp, int(r.choice([1448, 1448, 900, 300, 120])), s.rtt() / 4, push=False)
    s.tcp_close(s.client, srv, sp, dport)


def gen_https_tcp(s: _Session):
    _tls_exchange(s, 443, int(s.rng.integers(2, 9)))
    s.finish(0, "https_tcp", "Benign")


def gen_http_tcp(s: _Session):
    r, srv, sp = s.rng, s.server_ip(), 0
    sp = _eph(r)
    s.tcp_open(s.client, srv, sp, 80)
    s.tcp_data(s.client, srv, sp, 80, int(r.integers(180, 420)), 0.001)
    for _ in range(int(r.integers(1, 8))):
        s.tcp_data(srv, s.client, 80, sp, int(r.choice([1448, 1448, 700, 200])), s.rtt() / 3, push=False)
    s.tcp_close(s.client, srv, sp, 80)
    s.finish(0, "http_tcp", "Benign")


def gen_other_tcp(s: _Session):
    r, srv, port = s.rng, s.server_ip(), int(s.rng.choice([22, 993, 445, 3389, 5222, 8080]))
    sp = _eph(r)
    s.tcp_open(s.client, srv, sp, port)
    for _ in range(int(r.integers(6, 40))):
        s.tcp_data(s.client, srv, sp, port, int(r.integers(20, 160)), float(r.exponential(0.5)))
        s.tcp_data(srv, s.client, port, sp, int(r.integers(20, 400)), s.rtt() / 2)
    s.tcp_close(s.client, srv, sp, port)
    s.finish(0, "other_tcp", "Benign")


def gen_quic_udp(s: _Session):
    r, srv, sp = s.rng, s.server_ip(), 0
    sp = _eph(r)
    s.feed(IP(src=s.client, dst=srv) / UDP(sport=sp, dport=443) / Raw(b"i" * 1200), 0.001)       # Initial (padded)
    for _ in range(int(r.integers(2, 4))):
        s.feed(IP(src=srv, dst=s.client) / UDP(sport=443, dport=sp) / Raw(b"h" * int(r.choice([1200, 1252, 400]))), s.rtt() / 3)
    for _ in range(int(r.integers(10, 80))):
        s.feed(IP(src=s.client, dst=srv) / UDP(sport=sp, dport=443) / Raw(b"d" * int(r.integers(22, 110))), float(r.exponential(0.05)))
        s.feed(IP(src=srv, dst=s.client) / UDP(sport=443, dport=sp) / Raw(b"d" * int(r.choice([1252, 1252, 800, 70, 40]))), s.rtt() / 5)
    s.finish(0, "quic_udp", "Benign")


def gen_ntp_udp(s: _Session):
    r, srv, sp = s.rng, str(s.rng.choice(["129.6.15.28", "162.159.200.1", "216.239.35.0"])), 123
    s.feed(IP(src=s.client, dst=srv) / UDP(sport=123, dport=123) / Raw(b"n" * 48), 0.001)
    s.feed(IP(src=srv, dst=s.client) / UDP(sport=123, dport=123) / Raw(b"n" * 48), s.rtt())
    s.finish(0, "ntp_udp", "Benign")


def gen_dhcp_udp(s: _Session):
    s.feed(IP(src="0.0.0.0", dst="255.255.255.255") / UDP(sport=68, dport=67) / Raw(b"d" * 300), 0.001)  # nosec B104 - DHCP client source address, not a socket bind
    s.feed(IP(src="192.168.1.1", dst="255.255.255.255") / UDP(sport=67, dport=68) / Raw(b"o" * 300), s.rtt())
    s.finish(0, "dhcp_udp", "Benign")


def gen_mdns_udp(s: _Session):
    r = s.rng
    s.feed(IP(src=s.client, dst="224.0.0.251") / UDP(sport=5353, dport=5353) / Raw(b"m" * int(r.integers(30, 90))), 0.001)
    s.feed(IP(src=s.client, dst="224.0.0.251") / UDP(sport=5353, dport=5353) / Raw(b"m" * int(r.integers(30, 90))), 1.0)
    s.finish(0, "mdns_udp", "Benign")


def gen_ssdp_udp(s: _Session):
    s.feed(IP(src=s.client, dst="239.255.255.250") / UDP(sport=_eph(s.rng), dport=1900) / Raw(b"s" * 130), 0.001)
    s.finish(0, "ssdp_udp", "Benign")


def gen_icmp_ping(s: _Session):
    r, srv = s.rng, s.server_ip()
    for i in range(int(r.integers(2, 6))):
        s.feed(IP(src=s.client, dst=srv) / ICMP(type=8, id=7, seq=i) / Raw(b"p" * 32), 1.0)
        s.feed(IP(src=srv, dst=s.client) / ICMP(type=0, id=7, seq=i) / Raw(b"p" * 32), s.rtt())
    s.finish(0, "icmp_ping", "Benign")


def gen_igmp(s: _Session):
    for _ in range(int(s.rng.integers(1, 4))):
        s.feed(IP(src="192.168.1.1", dst="224.0.0.1", proto=2, ttl=1) / IGMP(type=0x11, mrcode=100, gaddr="0.0.0.0"), 0.5)  # nosec B104 - IGMP general-query group field
        s.feed(IP(src=s.client, dst="224.0.0.251", proto=2, ttl=1) / IGMP(type=0x16, gaddr="224.0.0.251"), 0.2)
    s.finish(0, "igmp", "Benign")


# ======================================================================== attack generators
def _syn(s: _Session, src, dst, sport, dport, bare: bool, dt: float):
    opts = [] if bare else [("MSS", 1460)]
    s.feed(IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags="S", options=opts), dt)


def gen_syn_scan_nmap(s: _Session):
    r, tgt, ports = s.rng, f"192.168.{int(s.rng.integers(0, 4))}.{int(s.rng.integers(2, 250))}", s.rng.permutation(np.arange(1, 1025))[:int(s.rng.integers(30, 80))]
    sp = int(r.integers(40000, 60000))
    for p in ports:
        _syn(s, s.client, tgt, sp, int(p), False, float(r.exponential(0.002)))
    s.finish(1, "syn_scan_nmap", "Scan")


def gen_syn_scan_bare(s: _Session):
    r, tgt, ports = s.rng, f"192.168.{int(s.rng.integers(0, 4))}.{int(s.rng.integers(2, 250))}", np.arange(1, 1 + int(s.rng.integers(30, 80)))
    for p in ports:
        _syn(s, s.client, tgt, int(r.integers(1024, 65535)), int(p), True, float(r.exponential(0.001)))
    s.finish(1, "syn_scan_bare", "Scan")


def gen_port_sweep(s: _Session):
    r, port = s.rng, int(s.rng.choice([22, 80, 443, 445, 3389]))
    base = int(r.integers(0, 4))
    for h in range(2, 2 + int(r.integers(30, 80))):
        _syn(s, s.client, f"192.168.{base}.{h}", int(r.integers(1024, 65535)), port, bool(r.integers(0, 2)), float(r.exponential(0.002)))
    s.finish(1, "port_sweep", "Scan")


def gen_syn_flood(s: _Session):
    r, tgt, port = s.rng, f"192.168.1.{int(s.rng.integers(2, 250))}", int(s.rng.choice([80, 443]))
    for _ in range(int(r.integers(80, 200))):
        src = f"{int(r.integers(1, 223))}.{int(r.integers(0, 255))}.{int(r.integers(0, 255))}.{int(r.integers(1, 255))}"
        _syn(s, src, tgt, int(r.integers(1024, 65535)), port, True, float(r.exponential(0.0005)))
    s.finish(1, "syn_flood", "DoS")


def gen_udp_flood(s: _Session):
    r, tgt = s.rng, f"192.168.1.{int(s.rng.integers(2, 250))}"
    sp, dp = int(r.integers(1024, 65535)), int(r.choice([53, 123, 19, 80, 5060]))
    for _ in range(int(r.integers(150, 400))):
        s.feed(IP(src=s.client, dst=tgt) / UDP(sport=sp, dport=dp) / Raw(b"f" * int(r.choice([512, 1024, 1400]))), float(r.exponential(0.0004)))
    s.finish(1, "udp_flood", "DoS")


def gen_icmp_sweep(s: _Session):
    r, base = s.rng, int(s.rng.integers(0, 4))
    for h in range(2, 2 + int(r.integers(30, 90))):
        s.feed(IP(src=s.client, dst=f"192.168.{base}.{h}") / ICMP(type=8, id=9, seq=h) / Raw(b"s" * 8), float(r.exponential(0.003)))
    s.finish(1, "icmp_sweep", "Scan")


def gen_brute_force_like(s: _Session):
    r, tgt, port = s.rng, f"192.168.1.{int(s.rng.integers(2, 250))}", int(s.rng.choice([22, 21, 3389]))
    for _ in range(int(r.integers(10, 40))):
        sp = _eph(r)
        s.tcp_open(s.client, tgt, sp, port)
        s.tcp_data(s.client, tgt, sp, port, int(r.integers(20, 60)), 0.01)
        s.tcp_data(tgt, s.client, port, sp, int(r.integers(30, 80)), 0.02)
        s.tcp_close(s.client, tgt, sp, port)
        s.finish(1, "brute_force_like", "BruteForce")
        s.pause(float(r.exponential(0.3)))


GENERATORS: Dict[str, Callable[[_Session], None]] = {
    "dns_udp": gen_dns_udp, "dns_tcp": gen_dns_tcp, "https_tcp": gen_https_tcp, "http_tcp": gen_http_tcp,
    "other_tcp": gen_other_tcp, "quic_udp": gen_quic_udp, "ntp_udp": gen_ntp_udp, "dhcp_udp": gen_dhcp_udp,
    "mdns_udp": gen_mdns_udp, "ssdp_udp": gen_ssdp_udp, "icmp_ping": gen_icmp_ping, "igmp": gen_igmp,
    "syn_scan_nmap": gen_syn_scan_nmap, "syn_scan_bare": gen_syn_scan_bare, "port_sweep": gen_port_sweep,
    "syn_flood": gen_syn_flood, "udp_flood": gen_udp_flood, "icmp_sweep": gen_icmp_sweep,
    "brute_force_like": gen_brute_force_like,
}

DEFAULT_COUNTS = {  # generator invocations per session (a "unit" can yield several flows for attacks)
    "dns_udp": 60, "dns_tcp": 6, "https_tcp": 60, "http_tcp": 15, "other_tcp": 10, "quic_udp": 25,
    "ntp_udp": 6, "dhcp_udp": 4, "mdns_udp": 10, "ssdp_udp": 6, "icmp_ping": 6, "igmp": 6,
    "syn_scan_nmap": 4, "syn_scan_bare": 4, "port_sweep": 3, "syn_flood": 2, "udp_flood": 3,
    "icmp_sweep": 3, "brute_force_like": 2,
}


def generate(seed: int = 0, n_sessions: int = 3, counts: Optional[Dict[str, int]] = None,
             floor: int = 52, classes: Optional[List[str]] = None, proc_cost_s: float = 0.0,
             proc_jitter_s: float = 0.0, extractor_factory=None, clock_modules=None) -> pd.DataFrame:
    """Return a DataFrame: FEATURE_COLS + meta (_label, _traffic_class, _family, _session, _synthetic, _timestamp ...).

    The generator is deterministic for a given (seed, counts, classes); `floor` only changes how the
    extractor reports the TCP minimum packet length, so floor on/off datasets are directly comparable.

    proc_cost_s / proc_jitter_s model processing lag (see module docstring). `extractor_factory(floor)` lets a caller
    plug in a different extractor implementation (used to compare a proposed extractor against the current one) and
    `clock_modules` lists the modules whose `time` attribute must be replaced by the virtual clock."""
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    use = classes or list(GENERATORS)
    master = np.random.default_rng(seed)
    clock_modules = list(clock_modules) if clock_modules else [_fx]
    saved = [m.time for m in clock_modules]
    all_records: List[dict] = []
    try:
        for si in range(n_sessions):
            srng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
            os_name = ["win", "linux", "mac"][si % 3]
            sess = _Session(srng, f"synthetic-s{si}", floor, os_name, t0=1_000_000.0 + si * 86_400.0,
                            extractor_factory=extractor_factory, proc_cost_s=proc_cost_s, proc_jitter_s=proc_jitter_s)
            for m in clock_modules:
                m.time = sess.clock
            schedule = [c for c in use for _ in range(int(counts.get(c, 0)))]
            srng.shuffle(schedule)
            for cls in schedule:
                GENERATORS[cls](sess)
                sess.pause(float(srng.exponential(2.0)))
            all_records.extend(sess.records)
    finally:
        for m, s in zip(clock_modules, saved):
            m.time = s
    df = pd.DataFrame(all_records)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df.sort_values("_timestamp").reset_index(drop=True)
