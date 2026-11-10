"""ARP anomaly module: detections, false-positive controls, and isolation from the current pipeline."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from modules.arp_monitor import ArpEvent, ArpMonitor, is_enabled

GW, GW_MAC = "192.168.1.1", "02:aa:aa:aa:aa:01"   # locally administered unicast MACs (multicast bit clear)
HOST, HOST_MAC = "192.168.1.50", "02:bb:bb:bb:bb:02"
ATTACKER = "02:cc:cc:cc:cc:03"
ROOT = Path(__file__).resolve().parent.parent


def req(ts, sip, smac, tip):
    return ArpEvent(ts, "request", sip, smac, tip)


def rep(ts, sip, smac, tip, tmac="00:11:22:33:44:55"):
    return ArpEvent(ts, "reply", sip, smac, tip, tmac)


def kinds(alerts):
    return [a.kind for a in alerts]


def find(alerts, kind):
    return next(a for a in alerts if a.kind == kind)


def test_normal_request_reply_exchange_raises_nothing():
    m = ArpMonitor(gateway_ips={GW})
    assert m.observe(req(0, HOST, HOST_MAC, GW)) == []
    assert m.observe(rep(0.01, GW, GW_MAC, HOST, HOST_MAC)) == []
    assert m.observe(req(30, HOST, HOST_MAC, GW)) == [] and m.observe(rep(30.01, GW, GW_MAC, HOST, HOST_MAC)) == []
    assert m.snapshot()[GW] == GW_MAC


def test_arp_spoof_of_a_host_is_flagged_high():
    m = ArpMonitor()
    m.observe(rep(0, HOST, HOST_MAC, GW))
    a = m.observe(rep(5, HOST, ATTACKER, GW))
    spoof = find(a, "IP_MAC_CHANGE")
    assert spoof.severity == "high" and spoof.details == {"previous_mac": HOST_MAC, "new_mac": ATTACKER}


def test_gateway_mac_change_is_critical():
    m = ArpMonitor(gateway_ips={GW})
    m.observe(rep(0, GW, GW_MAC, HOST))
    a = m.observe(rep(2, GW, ATTACKER, HOST))
    assert find(a, "GATEWAY_MAC_CHANGE").severity == "critical" and "IP_MAC_CHANGE" not in kinds(a)


def test_static_binding_catches_a_spoof_on_the_very_first_packet():
    m = ArpMonitor(static_bindings={GW: GW_MAC})
    assert "IP_MAC_CHANGE" in kinds(m.observe(rep(0, GW, ATTACKER, HOST)))


def test_trusted_mac_is_not_flagged_for_failover_or_proxy_arp():
    m = ArpMonitor(trusted_macs={ATTACKER}, max_ips_per_mac=2)
    m.observe(rep(0, GW, GW_MAC, HOST))
    assert not {"IP_MAC_CHANGE", "GATEWAY_MAC_CHANGE"} & set(kinds(m.observe(rep(1, GW, ATTACKER, HOST))))  # VRRP-style move
    for i in range(10):
        assert "MAC_MANY_IPS" not in kinds(m.observe(rep(2 + i * 0.1, f"10.0.0.{i}", ATTACKER, HOST)))


def test_one_mac_claiming_many_ips_is_flagged():
    m = ArpMonitor(max_ips_per_mac=5)
    alerts = []
    for i in range(10):
        alerts += m.observe(rep(i * 0.1, f"192.168.1.{100 + i}", ATTACKER, HOST))
    assert "MAC_MANY_IPS" in kinds(alerts)


def test_arp_sweep_and_gratuitous_flood():
    m = ArpMonitor(max_sweep_targets=10, max_gratuitous=3)
    alerts = []
    for i in range(30):
        alerts += m.observe(req(i * 0.05, HOST, HOST_MAC, f"192.168.1.{i + 1}"))
    assert "ARP_SWEEP" in kinds(alerts)
    g = []
    for i in range(8):
        g += m.observe(ArpEvent(100 + i * 0.1, "request", "192.168.1.77", "02:dd:dd:dd:dd:04", "192.168.1.77"))
    assert "GRATUITOUS_FLOOD" in kinds(g)


def test_dhcp_probe_and_matched_replies_are_not_alerts():
    m = ArpMonitor()
    assert m.observe(req(0, "0.0.0.0", HOST_MAC, "192.168.1.60")) == []      # duplicate-address detection
    m.observe(req(1, HOST, HOST_MAC, GW))
    assert "UNSOLICITED_REPLY" not in kinds(m.observe(rep(1.01, GW, GW_MAC, HOST, HOST_MAC)))


def test_unsolicited_reply_is_low_severity_and_invalid_senders_are_high():
    m = ArpMonitor()
    a = m.observe(rep(50, GW, GW_MAC, HOST, HOST_MAC))
    assert kinds(a) == ["UNSOLICITED_REPLY"] and a[0].severity == "low"
    bad = m.observe(rep(60, "1.2.3.4", "ff:ff:ff:ff:ff:ff", HOST))
    assert kinds(bad) == ["INVALID_SENDER"] and bad[0].severity == "high"
    assert kinds(m.observe(rep(62, "1.2.3.5", "01:00:5e:00:00:01", HOST))) == ["INVALID_SENDER"]   # multicast sender MAC
    assert kinds(m.observe(rep(61, "0.0.0.0", HOST_MAC, HOST))) == ["INVALID_SENDER"]


def test_alerts_are_deduplicated_inside_the_window():
    m = ArpMonitor(window_s=10)
    m.observe(rep(0, HOST, HOST_MAC, GW))
    first = m.observe(rep(1, HOST, ATTACKER, GW))
    again = m.observe(rep(2, HOST, ATTACKER, GW))
    assert kinds(first).count("IP_MAC_CHANGE") == 1 and "IP_MAC_CHANGE" not in kinds(again)


def test_a_failing_alert_handler_never_breaks_monitoring():
    m = ArpMonitor(on_alert=lambda a: 1 / 0)
    m.observe(rep(0, HOST, HOST_MAC, GW))
    assert "IP_MAC_CHANGE" in kinds(m.observe(rep(1, HOST, ATTACKER, GW)))


def test_memory_is_bounded():
    m = ArpMonitor(max_table=100)
    for i in range(1000):
        m.observe(rep(i, f"10.{i // 250}.{i % 250}.1", f"02:00:00:00:{i // 256:02x}:{i % 256:02x}", HOST))
    assert len(m.snapshot()) <= 1000 and len(m.bindings) < 1000


# ──────────────────────────────────────────────────────────────── isolation from the current pipeline
def test_module_imports_only_the_standard_library():
    code = (
        "import sys; before=set(sys.modules);"
        "import modules.arp_monitor;"
        "new=set(sys.modules)-before;"
        "bad=[m for m in new if m.split('.')[0] in ('app','flow_extractor','scapy','torch','xgboost','sklearn',"
        "'modules') and m!='modules' and m!='modules.arp_monitor'];"
        "print('BAD:',bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True).stdout
    assert "BAD: []" in out, out


def test_nothing_in_the_pipeline_imports_the_arp_monitor():
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", "modules/arp_monitor.py", "legacy/", ".venv")):
            continue
        if re.search(r"^\s*(from|import)\s+modules\.arp_monitor|arp_monitor", path.read_text(encoding="utf-8", errors="ignore"), re.M):
            offenders.append(rel)
    assert offenders == [], f"ARP monitor must stay unwired from the pipeline, found: {offenders}"


def test_flow_pipeline_still_ignores_arp_and_disabled_by_default(monkeypatch):
    from scapy.all import ARP, Ether
    from flow_extractor import FlowExtractor
    ex = FlowExtractor()
    ex.add_packet(Ether() / ARP(op=1, psrc=HOST, hwsrc=HOST_MAC, pdst=GW))
    assert ex.active_flow_count() == 0 and ex.pkt_errors == 0       # dropped cleanly, no error, no flow
    monkeypatch.delenv("CYBERGRAPH_ARP_MONITOR", raising=False)
    assert is_enabled() is False
    monkeypatch.setenv("CYBERGRAPH_ARP_MONITOR", "1")
    assert is_enabled() is True


def test_monitor_never_touches_the_firewall_or_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import subprocess as sp
    calls = []
    monkeypatch.setattr(sp, "run", lambda *a, **k: calls.append(a))
    m = ArpMonitor(gateway_ips={GW})
    m.observe(rep(0, GW, GW_MAC, HOST))
    m.observe(rep(1, GW, ATTACKER, HOST))
    assert calls == [] and list(tmp_path.iterdir()) == []
