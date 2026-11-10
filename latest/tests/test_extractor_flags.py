"""Regression tests for the extractor flags (tcp_close_mode, timestamp_source) and the lock.

The reference is tests/golden/flow_extractor_original.py: a frozen, verbatim copy of the extractor BEFORE the flags.
With default flags the current extractor must be bit-identical to it. The new modes are tested for: TCP teardown (no orphan
flows), RST, half-closed connections, port reuse, late stragglers, capture timestamps (invariance under processing delay,
fallbacks, out-of-order, replay of old captures), thread safety, and the diagnostics metadata."""
import importlib.util
import inspect
import threading
import time as real_time
from pathlib import Path

import numpy as np
import pytest
from scapy.all import IP, TCP, UDP, Ether, Raw

from evaluation.synthetic_traffic import generate
from train_xgboost_ae import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
PROP = flow_extractor_current = __import__("flow_extractor")           # the extractor under test (production module)
_spec = importlib.util.spec_from_file_location("flow_extractor_original",
                                               ROOT / "tests" / "golden" / "flow_extractor_original.py")
PROD = importlib.util.module_from_spec(_spec)                          # frozen reference (the pre-flag behaviour)
_spec.loader.exec_module(PROD)

C, S = "192.168.1.50", "93.184.216.34"


class Clock:
    """Virtual wall clock injected into the extractor module."""
    def __init__(self, now=1_000_000.0):
        self.now = now

    def time(self):
        return self.now

    def __getattr__(self, item):
        return getattr(real_time, item)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(PROP, "time", c)
    monkeypatch.setattr(PROD, "time", c)
    return c


def pkt(src, dst, sport, dport, flags, t=None, payload=b"", seq=1):
    p = Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags, seq=seq) / Raw(payload)
    if t is not None:
        p.time = t
    return p


def teardown_stream(sp=40000, t0=100.0):
    """SYN, SYN-ACK, ACK, data both ways, then client FIN-ACK, server FIN-ACK, final ACK (capture times t0 + ...)."""
    return [
        pkt(C, S, sp, 443, "S", t0), pkt(S, C, 443, sp, "SA", t0 + .02), pkt(C, S, sp, 443, "A", t0 + .03),
        pkt(C, S, sp, 443, "PA", t0 + .04, b"x" * 300, seq=2), pkt(S, C, 443, sp, "PA", t0 + .08, b"y" * 900, seq=2),
        pkt(C, S, sp, 443, "FA", t0 + .10, seq=302), pkt(S, C, 443, sp, "FA", t0 + .12, seq=902),
        pkt(C, S, sp, 443, "A", t0 + .13, seq=303),
    ]


def feed(ex, packets, clock=None, wall_step=None):
    out = []
    for p in packets:
        if clock is not None and wall_step is not None:
            clock.now += wall_step
        r = ex.add_packet(p)
        if r:
            out.append(r)
    return out


# ───────────────────────────────────────────────────── default flags == production
def test_default_flags_are_bit_identical_to_the_frozen_original():
    counts = {"dns_udp": 6, "https_tcp": 6, "quic_udp": 3, "igmp": 2, "syn_scan_bare": 2, "udp_flood": 1, "brute_force_like": 1}
    a = generate(seed=3, n_sessions=2, counts=counts, extractor_factory=lambda f: PROD.FlowExtractor(),
                 clock_modules=[PROD])
    b = generate(seed=3, n_sessions=2, counts=counts, extractor_factory=lambda f: PROP.FlowExtractor(min_pkt_floor_tcp=f),
                 clock_modules=[PROP])
    cols = FEATURE_COLS + ["_timestamp", "_pkt_count", "_byte_count", "_src_ip", "_dst_ip", "_src_port", "_dst_port"]
    assert len(a) == len(b) > 0
    assert a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))


def _random_stream(rng, n):
    """Mixed packets over a SMALL address/port space so flows collide, reopen, close and expire."""
    from scapy.all import ICMP
    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    out = []
    for _ in range(n):
        src, dst = rng.choice(hosts, 2, replace=False)
        kind = rng.integers(0, 6)
        sp, dp = int(rng.integers(1000, 1004)), int(rng.choice([53, 80, 443, 1001]))
        if kind <= 2:
            flags = "".join(rng.choice(list("FSRPAUEC"), int(rng.integers(1, 4)), replace=False))
            p = IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags=flags, seq=int(rng.integers(0, 50)),
                                           window=int(rng.integers(0, 65535))) / Raw(b"x" * int(rng.integers(0, 200)))
        elif kind <= 4:
            p = IP(src=src, dst=dst) / UDP(sport=sp, dport=dp) / Raw(rng.bytes(int(rng.integers(0, 120))))
        else:
            p = IP(src=src, dst=dst) / ICMP(type=int(rng.choice([0, 8, 3]))) / Raw(b"p" * 16)
        out.append((Ether() / p, float(rng.choice([0.0, 0.0005, 0.01, 0.3, 4.0, 12.0])), bool(rng.integers(0, 12) == 0)))
    return out


@pytest.mark.parametrize("seed", range(12))
def test_randomised_streams_default_flags_match_the_frozen_original_exactly(seed, monkeypatch):
    """Same packets, same virtual clock, interleaved flush_expired(): every exported record must be identical."""
    stream = _random_stream(np.random.default_rng(seed), 250)
    results = []
    for mod in (PROD, PROP):
        clk = Clock()
        monkeypatch.setattr(mod, "time", clk)
        ex = mod.FlowExtractor()
        exported = []
        for pk, dt, do_flush in stream:
            clk.now += dt
            r = ex.add_packet(pk)
            if r:
                exported.append(r)
            if do_flush:
                exported.extend(ex.flush_expired())
        clk.now += 100
        exported.extend(ex.flush_expired())
        results.append((exported, ex.ipv4_packets_seen, ex.pkt_errors))
    (ra, ia, ea), (rb, ib, eb) = results
    strip = lambda recs: [{k: v for k, v in r.items() if k not in ("_close_reason", "_midflow")} for r in recs]  # noqa: E731
    assert (ia, ea) == (ib, eb) and len(ra) == len(rb) > 0
    assert strip(ra) == strip(rb)


def test_defaults_keep_todays_behaviour_until_the_flags_are_switched_deliberately():
    sig = inspect.signature(PROP.FlowExtractor.__init__).parameters
    assert sig["tcp_close_mode"].default == "legacy" and sig["timestamp_source"].default == "wall"
    assert sig["min_pkt_floor_tcp"].default == 52 and sig["tcp_linger_s"].default == 2.0
    import live_capture
    src = inspect.getsource(live_capture)
    assert "FlowExtractor()" in src and "tcp_close_mode" not in src and "timestamp_source" not in src   # nothing enables them yet


def test_invalid_modes_are_rejected():
    with pytest.raises(ValueError):
        PROP.FlowExtractor(tcp_close_mode="bogus")
    with pytest.raises(ValueError):
        PROP.FlowExtractor(timestamp_source="bogus")


# ───────────────────────────────────────────────────── TCP teardown / orphan flows
def test_legacy_mode_reproduces_the_orphan_flows(clock):
    ex = PROP.FlowExtractor()
    done = feed(ex, teardown_stream())
    flows = done + ex.drain_all()
    assert len(flows) == 3                                         # main flow + 2 orphans (today's behaviour)
    assert sorted(int(f["IN_PKTS"] + f["OUT_PKTS"]) for f in flows)[:2] == [1, 1]


def test_linger_mode_keeps_the_whole_connection_in_one_flow(clock):
    ex = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    flows = feed(ex, teardown_stream()) + ex.drain_all()
    assert len(flows) == 1                                         # no orphans
    f = flows[0]
    assert f["IN_PKTS"] == 5 and f["OUT_PKTS"] == 3 and f["_close_reason"] == "fin_complete"
    assert int(f["CLIENT_TCP_FLAGS"]) & 1 and int(f["SERVER_TCP_FLAGS"]) & 1      # both FINs, like the training data
    assert f["_midflow"] is False and ex.stray_packets_swallowed == 0


def test_clean_teardown_is_exported_immediately_without_waiting_for_the_linger_timer(clock):
    ex = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    done = feed(ex, teardown_stream())
    assert len(done) == 1 and ex.active_flow_count() == 0          # returned by add_packet at the final ACK


def test_late_stragglers_after_teardown_do_not_become_orphans(clock):
    ex = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    done = feed(ex, teardown_stream())
    late = feed(ex, [pkt(S, C, 443, 40000, "A", 100.2), pkt(C, S, 40000, 443, "R", 100.25)])
    assert late == [] and ex.active_flow_count() == 0 and ex.stray_packets_swallowed == 2 and len(done) == 1
    clock.now += 5                                                 # tombstone expires; a NEW connection is accepted
    feed(ex, [pkt(C, S, 40000, 443, "S", 200.0)])
    assert ex.active_flow_count() == 1


def test_new_syn_inside_the_tombstone_window_is_not_swallowed(clock):
    ex = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    feed(ex, teardown_stream())
    feed(ex, [pkt(C, S, 40000, 443, "S", 100.3)])                  # immediate port reuse
    assert ex.active_flow_count() == 1


def test_half_closed_connection_is_exported_by_the_linger_timer(clock):
    ex = PROP.FlowExtractor(tcp_close_mode="linger", tcp_linger_s=2.0, timestamp_source="capture")
    feed(ex, teardown_stream()[:6])                                # server never answers the FIN
    assert ex.flush_expired() == [] and ex.active_flow_count() == 1
    clock.now += 2.5
    out = ex.flush_expired()
    assert len(out) == 1 and out[0]["_close_reason"] == "linger" and ex.active_flow_count() == 0


def test_rst_closes_after_the_linger_and_swallows_trailing_rsts(clock):
    ex = PROP.FlowExtractor(tcp_close_mode="linger", tcp_linger_s=1.0, timestamp_source="capture")
    feed(ex, [pkt(C, S, 41000, 80, "S", 10.0), pkt(S, C, 80, 41000, "SA", 10.02), pkt(C, S, 41000, 80, "A", 10.03),
              pkt(S, C, 80, 41000, "RA", 10.2), pkt(C, S, 41000, 80, "R", 10.21)])     # RST then RST from the other side
    clock.now += 1.5
    out = ex.flush_expired()
    assert len(out) == 1 and int(out[0]["TCP_FLAGS"]) & 4 and ex.active_flow_count() == 0


def test_port_reuse_on_a_closing_flow_finishes_the_old_connection_first(clock):
    ex = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    feed(ex, teardown_stream()[:6])                                # first FIN only, flow is closing
    feed(ex, [pkt(C, S, 40000, 443, "S", 101.0)])                  # new SYN on the same 5-tuple
    flushed = ex.flush_expired()
    assert len(flushed) == 1 and flushed[0]["_close_reason"] == "port_reuse" and ex.active_flow_count() == 1


def test_capture_started_mid_connection_is_flagged_not_dropped(clock):
    ex = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    feed(ex, [pkt(C, S, 42000, 443, "PA", 5.0, b"z" * 100), pkt(S, C, 443, 42000, "A", 5.01)])
    out = ex.drain_all()
    assert len(out) == 1 and out[0]["_midflow"] is True


def test_udp_icmp_and_idle_expiry_are_unchanged(clock):
    ex = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    feed(ex, [Ether() / IP(src=C, dst="8.8.8.8") / UDP(sport=5555, dport=53) / Raw(b"q" * 30)])
    assert ex.flush_expired() == []
    clock.now += 11
    out = ex.flush_expired()
    assert len(out) == 1 and out[0]["_close_reason"] == "idle_timeout"


# ───────────────────────────────────────────────────── timestamps
def test_capture_timestamps_define_all_timing_features_even_if_processing_is_slow(clock):
    ex = PROP.FlowExtractor(timestamp_source="capture")
    packets = [pkt(C, S, 43000, 9999, "PA", 100.0 + i * 0.010, b"x" * 100, seq=i * 100) for i in range(6)]
    for p in packets:
        clock.now += 0.5                                           # processing takes 500 ms per packet (wall clock)
        ex.add_packet(p)
    f = ex.drain_all()[0]
    assert f["FLOW_DURATION_MILLISECONDS"] == pytest.approx(50.0, abs=1e-3)
    assert f["SRC_TO_DST_IAT_AVG"] == pytest.approx(10.0, abs=1e-3) and f["_timestamp"] == pytest.approx(100.0)


def test_wall_mode_still_measures_processing_time_as_today(clock):
    ex = PROP.FlowExtractor()                                      # default
    for i in range(6):
        clock.now += 0.5
        ex.add_packet(pkt(C, S, 43000, 9999, "PA", 100.0 + i * 0.010, b"x" * 100, seq=i * 100))
    assert ex.drain_all()[0]["FLOW_DURATION_MILLISECONDS"] == pytest.approx(2500.0, abs=1e-3)


def test_missing_or_bad_capture_timestamps_fall_back_to_the_wall_clock(clock):
    ex = PROP.FlowExtractor(timestamp_source="capture")
    p = pkt(C, S, 44000, 9999, "PA", None, b"x")
    for bad in (None, 0, -5.0, float("nan"), float("inf"), "garbage"):
        if bad is not None:
            p.time = bad
        clock.now += 0.1
        ex.add_packet(p)
    assert ex.capture_ts_fallbacks >= 5 and ex.pkt_errors == 0
    assert ex.drain_all()[0]["_timestamp"] >= 1_000_000.0          # wall-clock epoch, not garbage


def test_out_of_order_capture_timestamps_never_produce_negative_intervals(clock):
    ex = PROP.FlowExtractor(timestamp_source="capture")
    for i, t in enumerate([10.0, 10.5, 10.3, 10.9, 10.8]):
        ex.add_packet(pkt(C, S, 45000, 9999, "PA", t, b"x", seq=i))
    f = ex.drain_all()[0]
    assert f["SRC_TO_DST_IAT_MIN"] >= 0.0 and f["FLOW_DURATION_MILLISECONDS"] == pytest.approx(900.0, abs=1e-3)


def test_replaying_an_old_capture_still_expires_flows_by_wall_clock(clock):
    ex = PROP.FlowExtractor(timestamp_source="capture")
    ex.add_packet(pkt(C, S, 46000, 9999, "PA", 500_000_000.0, b"x"))            # capture from 1985, far before the wall clock
    assert ex.flush_expired() == []
    clock.now += 11
    out = ex.flush_expired()
    assert len(out) == 1 and out[0]["_timestamp"] == 500_000_000.0              # old capture time preserved in the record


def test_timing_features_do_not_depend_on_processing_delay_for_synthetic_traffic():
    kw = dict(seed=7, n_sessions=1, counts={"dns_udp": 8, "quic_udp": 3, "udp_flood": 2, "https_tcp": 3},
              extractor_factory=lambda f: PROP.FlowExtractor(min_pkt_floor_tcp=f, tcp_close_mode="linger",
                                                             timestamp_source="capture"), clock_modules=[PROP])
    fast = generate(**kw)
    slow = generate(proc_cost_s=0.004, proc_jitter_s=0.002, **kw)
    cols = ["FLOW_DURATION_MILLISECONDS", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_AVG_THROUGHPUT", "IN_PKTS"]
    assert fast[cols].reset_index(drop=True).equals(slow[cols].reset_index(drop=True))


# ───────────────────────────────────────────────────── thread safety
@pytest.mark.parametrize("mode", ["legacy", "linger"])
def test_concurrent_capture_and_flush_threads_do_not_raise(mode):
    ex = PROP.FlowExtractor(idle_timeout_s=0.0, active_timeout_s=0.0, tcp_close_mode=mode)
    stop, errors = threading.Event(), []

    def flusher():
        while not stop.is_set():
            try:
                ex.flush_expired()
            except Exception as exc:
                errors.append(exc)

    t = threading.Thread(target=flusher)
    t.start()
    end = real_time.time() + 1.5
    n = 0
    while real_time.time() < end:
        sp = 1024 + (n % 60000)
        for flags in ("S", "A", "FA"):
            ex.add_packet(Ether() / IP(src=C, dst=S) / TCP(sport=sp, dport=80, flags=flags))
        n += 1
    stop.set()
    t.join()
    assert errors == [] and ex.pkt_errors == 0 and n > 50


def test_drain_all_empties_the_extractor():
    ex = PROP.FlowExtractor()
    ex.add_packet(Ether() / IP(src=C, dst=S) / UDP(sport=1, dport=2) / Raw(b"a"))
    assert len(ex.drain_all()) == 1 and ex.active_flow_count() == 0 and ex.drain_all() == []


def test_the_model_features_are_unchanged_by_the_new_metadata():
    ex = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    f = feed(ex, teardown_stream())[0]
    assert all(c in f for c in FEATURE_COLS) and "_close_reason" in f
    assert np.isfinite([f[c] for c in FEATURE_COLS]).all()


# ───────────────────────────────────────────────────── Windows & Unix-style traffic tests
def test_windows_style_tcp_traffic_and_rst_ack(clock):
    """Windows TCP stack: TTL 128, win 64240, graceful FIN teardown (FA -> A -> FA -> A) and WinSock RST-ACK."""
    win_client, srv = "10.0.0.15", "10.0.0.1"
    sp = 51234
    dp = 445  # SMB port
    t0 = 200.0

    def win_pkt(src, dst, sport, dport, flags, t, payload=b"", seq=1):
        p = Ether() / IP(src=src, dst=dst, ttl=128) / TCP(sport=sport, dport=dport, flags=flags, window=64240, seq=seq) / Raw(payload)
        p.time = t
        return p

    # Standard Windows 4-step graceful teardown
    pkts = [
        win_pkt(win_client, srv, sp, dp, "S", t0),
        win_pkt(srv, win_client, dp, sp, "SA", t0 + 0.01),
        win_pkt(win_client, srv, sp, dp, "A", t0 + 0.02),
        win_pkt(win_client, srv, sp, dp, "PA", t0 + 0.03, b"smb_req", seq=2),
        win_pkt(srv, win_client, dp, sp, "PA", t0 + 0.05, b"smb_resp", seq=2),
        win_pkt(win_client, srv, sp, dp, "FA", t0 + 0.08, seq=10),
        win_pkt(srv, win_client, dp, sp, "A", t0 + 0.09, seq=10),
        win_pkt(srv, win_client, dp, sp, "FA", t0 + 0.10, seq=11),
        win_pkt(win_client, srv, sp, dp, "A", t0 + 0.11, seq=11),
    ]

    # In legacy mode: first FIN creates flow and exports immediately, remaining packets create orphans
    ex_legacy = PROP.FlowExtractor()
    out_legacy = feed(ex_legacy, pkts) + ex_legacy.drain_all()
    assert len(out_legacy) >= 2, "Legacy mode must create orphan flows on Windows teardown"

    # In linger mode: single consolidated bidirectional flow
    ex_linger = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    out_linger = feed(ex_linger, pkts) + ex_linger.drain_all()
    assert len(out_linger) == 1, "Linger mode must consolidate entire Windows teardown into exactly 1 flow"
    f = out_linger[0]
    assert f["TCP_WIN_MAX_IN"] == 64240
    assert f["_close_reason"] == "fin_complete"
    assert int(f["CLIENT_TCP_FLAGS"]) & 1 and int(f["SERVER_TCP_FLAGS"]) & 1


def test_unix_style_tcp_traffic_simultaneous_close_and_bare_rst(clock):
    """Unix/Linux TCP stack: TTL 64, win 29200, simultaneous close and bare RST."""
    linux_client, srv = "192.168.1.100", "192.168.1.1"
    sp = 48100
    dp = 80
    t0 = 300.0

    def unix_pkt(src, dst, sport, dport, flags, t, payload=b"", seq=1):
        p = Ether() / IP(src=src, dst=dst, ttl=64) / TCP(sport=sport, dport=dport, flags=flags, window=29200, seq=seq) / Raw(payload)
        p.time = t
        return p

    # Simultaneous close (both sides send FIN concurrently)
    pkts = [
        unix_pkt(linux_client, srv, sp, dp, "S", t0),
        unix_pkt(srv, linux_client, dp, sp, "SA", t0 + 0.005),
        unix_pkt(linux_client, srv, sp, dp, "A", t0 + 0.006),
        unix_pkt(linux_client, srv, sp, dp, "FA", t0 + 0.02, seq=2),
        unix_pkt(srv, linux_client, dp, sp, "FA", t0 + 0.022, seq=2),
        unix_pkt(linux_client, srv, sp, dp, "A", t0 + 0.025, seq=3),
        unix_pkt(srv, linux_client, dp, sp, "A", t0 + 0.026, seq=3),
    ]

    ex_linger = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    out = feed(ex_linger, pkts) + ex_linger.drain_all()
    assert len(out) == 1
    assert out[0]["TCP_WIN_MAX_IN"] == 29200
    assert int(out[0]["CLIENT_TCP_FLAGS"]) & 1 and int(out[0]["SERVER_TCP_FLAGS"]) & 1

    # Linux bare RST on abnormal abort
    sp2 = 48102
    rst_pkts = [
        unix_pkt(linux_client, srv, sp2, dp, "S", t0 + 1.0),
        unix_pkt(srv, linux_client, dp, sp2, "SA", t0 + 1.01),
        unix_pkt(linux_client, srv, sp2, dp, "R", t0 + 1.02),  # bare RST
    ]
    ex_rst = PROP.FlowExtractor(tcp_close_mode="linger", tcp_linger_s=1.0, timestamp_source="capture")
    feed(ex_rst, rst_pkts)
    clock.now += 2.0
    rst_out = ex_rst.flush_expired()
    assert len(rst_out) == 1
    assert int(rst_out[0]["TCP_FLAGS"]) & 4


def test_extractor_rollback_mechanism(monkeypatch):
    """When CYBERGRAPH_ROLLBACK_EXTRACTOR=1 is set, FlowExtractor forces legacy mode unconditionally."""
    monkeypatch.setenv("CYBERGRAPH_ROLLBACK_EXTRACTOR", "1")
    ex = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")
    assert ex._close_mode == "legacy"
    assert ex._ts_source == "wall"


def test_rollback_defaults_constant_and_factory():
    """Verify ROLLBACK_DEFAULTS is exposed and restores legacy behaviour."""
    assert hasattr(PROP, "ROLLBACK_DEFAULTS")
    assert PROP.ROLLBACK_DEFAULTS["tcp_close_mode"] == "legacy"
    assert PROP.ROLLBACK_DEFAULTS["timestamp_source"] == "wall"
    assert PROP.ROLLBACK_DEFAULTS["min_pkt_floor_tcp"] == 52
    assert PROP.ROLLBACK_DEFAULTS["tcp_linger_s"] == 2.0
    ex = PROP.FlowExtractor(**PROP.ROLLBACK_DEFAULTS)
    assert ex._close_mode == "legacy"
    assert ex._ts_source == "wall"


def test_windows_and_unix_sessions_synthetic_traffic_bit_identical():
    """Verify that multi-OS synthetic traffic (Windows bare ACKs, Linux timestamp options) is bit-identical between PROD and PROP."""
    counts = {"https_tcp": 3, "http_tcp": 3, "other_tcp": 3}
    df_prod = generate(seed=42, n_sessions=3, counts=counts, extractor_factory=lambda f: PROD.FlowExtractor(), clock_modules=[PROD])
    df_prop = generate(seed=42, n_sessions=3, counts=counts, extractor_factory=lambda f: PROP.FlowExtractor(min_pkt_floor_tcp=f), clock_modules=[PROP])
    cols = FEATURE_COLS + ["_timestamp", "_pkt_count", "_byte_count"]
    assert len(df_prod) == len(df_prop) > 0
    assert df_prod[cols].reset_index(drop=True).equals(df_prop[cols].reset_index(drop=True))


def test_windows_bare_ack_packet_floor_preserved_in_both_modes():
    """Ensure production default 52-byte TCP floor is preserved for bare Windows 40-byte ACKs across modes."""
    win_client, srv = "10.0.0.20", "10.0.0.1"
    sp, dp, t0 = 55000, 80, 500.0

    def bare_ack_stream():
        return [
            Ether() / IP(src=win_client, dst=srv, ttl=128) / TCP(sport=sp, dport=dp, flags="S", window=64240, seq=1),
            Ether() / IP(src=srv, dst=win_client, ttl=64) / TCP(sport=dp, dport=sp, flags="SA", window=29200, seq=1),
            # Bare 40-byte ACK (no TCP options)
            Ether() / IP(src=win_client, dst=srv, ttl=128) / TCP(sport=sp, dport=dp, flags="A", window=64240, seq=2),
            Ether() / IP(src=win_client, dst=srv, ttl=128) / TCP(sport=sp, dport=dp, flags="FA", window=64240, seq=2),
        ]

    # Default floor (52): MIN_IP_PKT_LEN must be 52.0 in both legacy and linger
    ex_leg = PROP.FlowExtractor()  # default min_pkt_floor_tcp=52
    out_leg = feed(ex_leg, bare_ack_stream())
    assert out_leg[0]["MIN_IP_PKT_LEN"] == 52.0

    ex_ling = PROP.FlowExtractor(tcp_close_mode="linger", timestamp_source="capture")  # default floor 52
    feed(ex_ling, bare_ack_stream())
    out_ling = ex_ling.drain_all()
    assert out_ling[0]["MIN_IP_PKT_LEN"] == 52.0

