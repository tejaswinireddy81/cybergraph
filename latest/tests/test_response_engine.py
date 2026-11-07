"""
Unit tests for response engine, FirewallController dry-run/enforce modes, rate-limiting, and IP whitelist protection.
"""

import pytest
from response.firewall import FirewallController
from response.response_engine import ResponsePolicyEngine


def test_firewall_controller_dry_run_vs_enforce():
    # Test dry-run mode initialization
    fw_dry = FirewallController(mode="dry-run")
    assert fw_dry.mode == "dry-run"
    assert fw_dry.dry_run is True
    assert fw_dry.enabled is False

    res_dry = fw_dry.block_ip("192.168.1.100", reason="Test block")
    assert res_dry["simulated"] is True
    assert res_dry["mode"] == "dry-run"

    # Test enforce mode set_mode
    fw_dry.set_mode("enforce")
    assert fw_dry.mode == "enforce"
    assert fw_dry.enabled is True
    assert fw_dry.dry_run is False


def test_firewall_whitelist_protection():
    fw = FirewallController(mode="enforce")

    # Loopback IP must be protected from being blocked
    res_local = fw.block_ip("127.0.0.1")
    assert res_local.get("suppressed") is True
    assert "whitelisted" in res_local.get("reason", "")
    assert fw.is_blocked("127.0.0.1") is False


def test_firewall_rate_limiting():
    fw = FirewallController(mode="dry-run", max_blocks_per_minute=3)

    # First 3 blocks should succeed
    for i in range(1, 4):
        res = fw.block_ip(f"192.168.1.{i}")
        assert res.get("suppressed") is not True

    # 4th block should be suppressed due to rate limiting
    res_4th = fw.block_ip("192.168.1.4")
    assert res_4th.get("suppressed") is True
    assert "Rate-limit" in res_4th.get("reason", "")


def test_response_policy_engine_severities(tmp_path):
    fw = FirewallController(mode="dry-run")
    engine = ResponsePolicyEngine(firewall=fw)

    # Low severity -> Logged, no alert, no firewall action
    res_low = engine.handle(src_ip="10.0.0.1", severity="Low")
    assert res_low["action"] == "logged"
    assert res_low["alert"] is False

    # Medium severity -> Alerted
    res_med = engine.handle(src_ip="10.0.0.2", severity="Medium")
    assert res_med["action"] == "alerted_increased_monitoring"
    assert res_med["alert"] is True

    # High severity -> Blacklisted
    res_high = engine.handle(src_ip="10.0.0.3", severity="High")
    assert res_high["action"] == "alerted_blacklisted"
    assert fw.is_blacklisted("10.0.0.3") is True

    # Critical severity (confidence verified >= 0.85) -> Blocked
    res_crit = engine.handle(src_ip="10.0.0.4", severity="Critical", confidence=0.92)
    assert res_crit["action"] == "blocked"
    assert fw.is_blocked("10.0.0.4") is True
