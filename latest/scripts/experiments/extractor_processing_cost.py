"""
scripts/experiments/extractor_processing_cost.py
================================================
How large is the processing delay that makes wall-clock timestamps differ from capture timestamps?

Part 1 (measured, offline, no capture): CPU cost of FlowExtractor.add_packet per packet on this machine.
Part 2 (illustrative): per-class error of duration / IAT / throughput for the production (wall-clock) extractor when each
        packet's processing takes {1x, 5x, 20x} the measured cost. The multipliers stand in for queueing behind a busy
        capture thread or Npcap batch delivery, which can only be measured with a real capture (lab sign-off pending).
The capture-timestamp extractor has exactly zero error by construction; that is asserted here.

    python scripts/experiments/extractor_processing_cost.py
"""
import importlib.util
import os
import sys
import time
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from scapy.all import IP, TCP, UDP, Ether, Raw  # noqa: E402

import flow_extractor as PROP  # noqa: E402   current extractor (has the tcp_close_mode / timestamp_source flags)
from evaluation.synthetic_traffic import generate  # noqa: E402

# frozen, verbatim copy of the extractor BEFORE the flags: the "as it was" reference
spec = importlib.util.spec_from_file_location("flow_extractor_original",
                                              os.path.join(ROOT, "tests", "golden", "flow_extractor_original.py"))
PROD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROD)


def measure_cost(n=20000):
    pkts = []
    for i in range(n):
        if i % 2:
            pkts.append(Ether() / IP(src="10.0.0.1", dst="8.8.8.8") / UDP(sport=40000 + i % 2000, dport=53) / Raw(b"q" * 30))
        else:
            pkts.append(Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=40000 + i % 2000, dport=443, flags="PA", seq=i) / Raw(b"x" * 400))
    ex = PROD.FlowExtractor()
    t0 = time.perf_counter()
    for p in pkts:
        ex.add_packet(p)
    return (time.perf_counter() - t0) / n


def aligned(df):
    key = ["_session", "_src_ip", "_src_port", "_dst_ip", "_dst_port", "PROTOCOL"]
    d = df.sort_values("_timestamp").copy()
    d["_k"] = d.groupby(key).cumcount()
    return d.set_index(key + ["_k"])


def main():
    cost = min(measure_cost() for _ in range(3))
    print(f"MEASURED extractor cost: {cost * 1e6:.0f} microseconds per packet (best of 3, this machine, excludes capture/dissection by sniff)")
    kw = dict(seed=11, n_sessions=2, counts={"dns_udp": 40, "quic_udp": 12, "https_tcp": 12, "ntp_udp": 6,
                                             "syn_scan_bare": 4, "port_sweep": 3, "syn_flood": 2, "udp_flood": 3, "icmp_sweep": 3})
    truth = aligned(generate(**kw))
    print(f"\nillustrative per-class error of the PRODUCTION (wall-clock) extractor, {len(truth)} flows; truth = capture timestamps")
    print(f"  {'class':18s}{'cost':>8s}{'duration med/p90 rel.err':>28s}{'throughput med rel.err':>26s}{'flows >10% off':>16s}")
    classes = ["dns_udp", "quic_udp", "https_tcp", "syn_scan_bare", "syn_flood", "udp_flood", "icmp_sweep"]
    for mult in (1, 5, 20):
        proc = cost * mult
        d = aligned(generate(proc_cost_s=proc, proc_jitter_s=proc * 0.5, **kw))
        j = truth.join(d, how="inner", lsuffix="_t", rsuffix="_d")
        e_dur = (j.FLOW_DURATION_MILLISECONDS_d - j.FLOW_DURATION_MILLISECONDS_t).abs() / np.maximum(j.FLOW_DURATION_MILLISECONDS_t, 1.0)
        e_tp = (j.SRC_TO_DST_AVG_THROUGHPUT_d - j.SRC_TO_DST_AVG_THROUGHPUT_t).abs() / np.maximum(j.SRC_TO_DST_AVG_THROUGHPUT_t, 1.0)
        for cls in classes:
            m = (j._traffic_class_t == cls).to_numpy()
            if m.any():
                print(f"  {cls:18s}{proc * 1e3:7.2f}ms{100 * e_dur[m].median():12.1f}% /{100 * e_dur[m].quantile(.9):7.1f}%{100 * e_tp[m].median():22.1f}%{100 * (e_dur[m] > .1).mean():15.1f}%")
        print()
    # the capture-timestamp extractor must be exact regardless of processing delay
    d = aligned(generate(proc_cost_s=cost * 20, proc_jitter_s=cost * 10, extractor_factory=lambda fl: PROP.FlowExtractor(
        min_pkt_floor_tcp=fl, timestamp_source="capture"), clock_modules=[PROP], **kw))
    j = truth.join(d, how="inner", lsuffix="_t", rsuffix="_d")
    exact = bool(np.allclose(j.FLOW_DURATION_MILLISECONDS_d, j.FLOW_DURATION_MILLISECONDS_t, atol=1e-6)
                 and np.allclose(j.SRC_TO_DST_IAT_AVG_d, j.SRC_TO_DST_IAT_AVG_t, atol=1e-6))
    print(f"capture-timestamp extractor under 20x processing delay: durations and IATs identical to truth = {exact}")


if __name__ == "__main__":
    main()
