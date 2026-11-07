"""
Unit tests for flow_extractor.py against known packet fixtures.
"""

import pytest
import time
from scapy.layers.inet import IP, TCP, UDP, ICMP
from flow_extractor import FlowExtractor, _parse_dns, _parse_ftp_code, _infer_l7, _safe_std


def test_parse_dns():
    # Invalid raw payload
    qid, qtype, ttl = _parse_dns(b"short")
    assert qid == 0 and qtype == 0 and ttl == 0


def test_parse_ftp_code():
    assert _parse_ftp_code(b"220 Welcome to FTP server\r\n") == 220
    assert _parse_ftp_code(b"530 Login incorrect\r\n") == 530
    assert _parse_ftp_code(b"invalid payload") == 0


def test_infer_l7():
    assert _infer_l7(12345, 80, 6) == 7    # HTTP
    assert _infer_l7(12345, 443, 6) == 91  # SSL/TLS
    assert _infer_l7(12345, 53, 17) == 5   # DNS
    assert _infer_l7(0, 0, 1) == 1         # ICMP


def test_safe_std():
    assert _safe_std([]) == 0.0
    assert _safe_std([5.0]) == 0.0
    assert round(_safe_std([10.0, 20.0]), 2) == 5.0


def test_flow_extractor_tcp_lifecycle():
    extractor = FlowExtractor(idle_timeout_s=5.0, active_timeout_s=30.0)

    # 1. SYN packet (Client -> Server)
    syn_pkt = IP(src="192.168.1.10", dst="192.168.1.1") / TCP(sport=54321, dport=80, flags="S", seq=1000)
    res = extractor.add_packet(syn_pkt)
    assert res is None
    assert extractor.active_flow_count() == 1

    # 2. SYN-ACK packet (Server -> Client)
    synack_pkt = IP(src="192.168.1.1", dst="192.168.1.10") / TCP(sport=80, dport=54321, flags="SA", seq=5000)
    res = extractor.add_packet(synack_pkt)
    assert res is None

    # 3. FIN packet (Client -> Server) closes flow
    fin_pkt = IP(src="192.168.1.10", dst="192.168.1.1") / TCP(sport=54321, dport=80, flags="FA", seq=1001)
    flow_dict = extractor.add_packet(fin_pkt)
    
    assert flow_dict is not None
    assert flow_dict["_src_ip"] == "192.168.1.10"
    assert flow_dict["_dst_ip"] == "192.168.1.1"
    assert flow_dict["L4_SRC_PORT"] == 54321.0
    assert flow_dict["L4_DST_PORT"] == 80.0
    assert flow_dict["PROTOCOL"] == 6.0
    assert flow_dict["IN_PKTS"] == 2.0
    assert flow_dict["OUT_PKTS"] == 1.0
    assert extractor.active_flow_count() == 0


def test_flow_extractor_timeout_flushing():
    extractor = FlowExtractor(idle_timeout_s=0.1, active_timeout_s=0.5)

    # Add UDP packet
    udp_pkt = IP(src="10.0.0.5", dst="8.8.8.8") / UDP(sport=1234, dport=53)
    extractor.add_packet(udp_pkt)
    assert extractor.active_flow_count() == 1

    # Wait for idle timeout
    time.sleep(0.2)
    flushed = extractor.flush_expired()
    
    assert len(flushed) == 1
    assert flushed[0]["_src_ip"] == "10.0.0.5"
    assert flushed[0]["_protocol"] == "UDP"
    assert flushed[0]["PROTOCOL"] == 17.0
    assert extractor.active_flow_count() == 0
