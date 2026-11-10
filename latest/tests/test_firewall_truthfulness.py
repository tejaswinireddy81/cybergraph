"""Regression tests: firewall executor must report what the OS actually did."""
from unittest import mock

from modules.firewall_response_executor import FirewallResponseExecutor


def _win_executor():
    fw = FirewallResponseExecutor(mode="enforce")
    fw.os_type = "windows"
    return fw


def test_unblock_reports_failure_when_rule_survives():
    fw = _win_executor()
    fw.active_os_firewall_rules["192.0.2.10"] = "x"
    failed = mock.Mock(returncode=1, stdout="", stderr="requires elevation")
    with mock.patch.object(fw, "_windows_rule_exists", return_value=True), \
         mock.patch("subprocess.run", return_value=failed):
        res = fw.execute_unblock("192.0.2.10")
    assert res["status"] == "UNBLOCK_FAILED"
    assert "192.0.2.10" in fw.active_os_firewall_rules  # still tracked for retry


def test_unblock_succeeds_when_rule_removed():
    fw = _win_executor()
    fw.active_os_firewall_rules["192.0.2.11"] = "x"
    ok = mock.Mock(returncode=0, stdout="Deleted 1 rule(s).", stderr="")
    with mock.patch.object(fw, "_windows_rule_exists", side_effect=[True, False, False]), \
         mock.patch("subprocess.run", return_value=ok):
        res = fw.execute_unblock("192.0.2.11")
    assert res["status"] == "UNBLOCKED_SUCCESS"
    assert "192.0.2.11" not in fw.active_os_firewall_rules


def test_elevated_block_not_reported_success_if_rule_missing():
    fw = _win_executor()
    denied = mock.Mock(returncode=1, stdout="", stderr="denied")
    launched = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch.object(fw, "_windows_rule_exists", return_value=False), \
         mock.patch("subprocess.run", side_effect=[denied, launched]):
        res = fw.execute_block("192.0.2.12", force_real=True)
    assert res["status"] == "ADMIN_PRIVILEGE_REQUIRED"
    assert "192.0.2.12" not in fw.active_os_firewall_rules


def test_verify_blocked_asks_the_os():
    fw = _win_executor()
    fw.active_os_firewall_rules["192.0.2.13"] = "x"
    with mock.patch.object(fw, "_windows_rule_exists", return_value=False):
        assert fw.verify_blocked("192.0.2.13")["real_os_firewall_blocked"] is False
