"""Firewall safety regression tests.

Guarantees under test:
  * dry-run (the default) NEVER executes an OS firewall command
  * real commands need explicit enforce mode AND an admin-initiated call
  * automated alert responses never execute real commands, even in enforce mode
  * the admin API requires confirmation before real blocking / enabling enforce mode
  * every block, unblock and mode change is written to the audit log
"""
import json
import re
import subprocess
from types import SimpleNamespace

import pytest

from app import app, get_admin_token
from modules.firewall_response_executor import FirewallResponseExecutor

FIREWALL_BINARIES = ("netsh", "iptables", "powershell")


@pytest.fixture
def os_calls(monkeypatch):
    """Record every firewall-related subprocess call (replaces the conftest fake)."""
    calls = []
    rules = set()

    def spy(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and cmd and str(cmd[0]).lower() in FIREWALL_BINARIES:
            calls.append(list(cmd))
            text = " ".join(map(str, cmd))
            m = re.search(r"""name="?([^"\s']+)""", text)
            name = m.group(1) if m else None
            low = text.lower()
            if "add rule" in low and name:
                rules.add(name)
            elif "delete rule" in low and name:
                rules.discard(name)
            elif "show rule" in low:
                if name in rules:
                    return SimpleNamespace(returncode=0, stdout="Rule Name: " + name, stderr="")
                return SimpleNamespace(returncode=1, stdout="No rules match the specified criteria.", stderr="")
            return SimpleNamespace(returncode=0, stdout="Ok.", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", spy)
    return calls


@pytest.fixture
def audit_path(tmp_path):
    return tmp_path / "audit.jsonl"


def make(mode, audit_path):
    fw = FirewallResponseExecutor(mode=mode, audit_log_path=str(audit_path))
    fw.os_type = "windows"
    return fw


def read_audit(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── mode defaults ──────────────────────────────────────────────────────────

def test_default_mode_is_dry_run(monkeypatch):
    monkeypatch.delenv("CYBERGRAPH_FIREWALL_MODE", raising=False)
    assert FirewallResponseExecutor().mode == "dry-run"
    assert FirewallResponseExecutor(enable_real_blocking=False).mode == "dry-run"


def test_invalid_env_mode_falls_back_to_dry_run(monkeypatch):
    monkeypatch.setenv("CYBERGRAPH_FIREWALL_MODE", "yolo")
    assert FirewallResponseExecutor().mode == "dry-run"


def test_env_can_select_enforce_explicitly(monkeypatch):
    monkeypatch.setenv("CYBERGRAPH_FIREWALL_MODE", "enforce")
    assert FirewallResponseExecutor().mode == "enforce"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        FirewallResponseExecutor(mode="maybe")


# ── dry-run never touches the OS ───────────────────────────────────────────

@pytest.mark.parametrize("kwargs", [{}, {"force_real": True}, {"allow_real": True}])
def test_dry_run_never_executes_os_command(os_calls, audit_path, kwargs):
    fw = make("dry-run", audit_path)
    res = fw.execute_block("203.0.113.7", **kwargs)
    assert os_calls == []
    assert res["status"] == "SIMULATED_BLOCK_RECORDED"
    assert res["execution_mode"] == "SIMULATED_MEMORY_ONLY"
    assert "203.0.113.7" not in fw.active_os_firewall_rules
    if kwargs:
        assert res["dry_run"] is True and res["real_block_requested"] is True


def test_dry_run_unblock_never_executes_os_command(os_calls, audit_path):
    fw = make("dry-run", audit_path)
    fw.execute_block("203.0.113.8", allow_real=True)
    res = fw.execute_unblock("203.0.113.8")
    assert os_calls == []
    assert res["status"] == "UNBLOCKED_SUCCESS"


def test_dry_run_verify_never_queries_os(os_calls, audit_path):
    fw = make("dry-run", audit_path)
    fw.execute_block("203.0.113.9")
    fw.verify_blocked("203.0.113.9")
    assert os_calls == []


# ── enforce mode requires admin intent ─────────────────────────────────────

def test_enforce_without_admin_intent_stays_simulated(os_calls, audit_path):
    """Automated alert responses call execute_block() without allow_real."""
    fw = make("enforce", audit_path)
    res = fw.execute_block("203.0.113.10")
    assert os_calls == []
    assert res["execution_mode"] == "SIMULATED_MEMORY_ONLY"


def test_enforce_with_admin_intent_executes_real_command(os_calls, audit_path):
    fw = make("enforce", audit_path)
    res = fw.execute_block("203.0.113.11", allow_real=True, actor="admin@test")
    assert any("add" in c and "rule" in c for c in os_calls)
    assert res["status"] == "REAL_FIREWALL_RULE_APPLIED"
    assert "203.0.113.11" in fw.active_os_firewall_rules


def test_protected_and_invalid_ips_never_reach_os(os_calls, audit_path):
    fw = make("enforce", audit_path)
    fw.add_protected_ip("192.168.1.1")
    assert fw.execute_block("192.168.1.1", allow_real=True)["execution_mode"] == "PROTECTED_WHITELIST"
    assert fw.execute_block("127.0.0.1", allow_real=True)["status"] == "ERROR"
    assert fw.execute_block("1.1.1.1;calc", allow_real=True)["status"] == "ERROR"
    assert os_calls == []


def test_cannot_leave_enforce_while_real_rules_active(os_calls, audit_path):
    fw = make("enforce", audit_path)
    fw.execute_block("203.0.113.12", allow_real=True)
    with pytest.raises(ValueError):
        fw.set_mode("dry-run")
    assert fw.mode == "enforce"
    fw.execute_unblock("203.0.113.12")
    assert fw.set_mode("dry-run")["mode"] == "dry-run"


# ── audit log ──────────────────────────────────────────────────────────────

def test_every_action_is_audited(os_calls, audit_path):
    fw = make("dry-run", audit_path)
    fw.execute_block("203.0.113.13", actor="admin@1.2.3.4")
    fw.execute_unblock("203.0.113.13", actor="admin@1.2.3.4")
    fw.set_mode("enforce", actor="admin@1.2.3.4")
    fw.execute_block("1.1.1.1;calc", actor="admin@1.2.3.4")  # rejected attempts are logged too
    rows = read_audit(audit_path)
    assert [r["action"] for r in rows] == ["BLOCK", "UNBLOCK", "MODE_CHANGE", "BLOCK"]
    assert all(r["actor"] == "admin@1.2.3.4" for r in rows)
    assert rows[0]["mode"] == "dry-run" and rows[0]["ip"] == "203.0.113.13"
    assert rows[2]["previous_mode"] == "dry-run" and rows[2]["new_mode"] == "enforce"
    assert rows[3]["status"] == "ERROR"
    assert all("timestamp" in r for r in rows)


def test_audit_failure_never_breaks_blocking(os_calls, tmp_path):
    fw = FirewallResponseExecutor(mode="dry-run", audit_log_path=str(tmp_path))  # a directory: unwritable
    assert fw.execute_block("203.0.113.14")["status"] == "SIMULATED_BLOCK_RECORDED"


# ── HTTP API ───────────────────────────────────────────────────────────────

@pytest.fixture
def admin(monkeypatch, tmp_path, os_calls):
    import app as main_app
    app.config["TESTING"] = True
    main_app.init_engines()
    fw = main_app.fw_executor
    saved = (fw.mode, dict(fw.active_os_firewall_rules), set(fw.simulated_acl_blocked_ips),
             fw.audit_log_path, fw.os_type)
    fw.mode = "dry-run"
    fw.active_os_firewall_rules.clear()
    fw.audit_log_path = str(tmp_path / "api_audit.jsonl")
    fw.os_type = "windows"
    headers = {"X-Admin-Token": get_admin_token()}
    with app.test_client() as client:
        yield SimpleNamespace(client=client, headers=headers, fw=fw, audit=tmp_path / "api_audit.jsonl")
    fw.mode, fw.os_type = saved[0], saved[4]
    fw.active_os_firewall_rules.clear()
    fw.active_os_firewall_rules.update(saved[1])
    fw.simulated_acl_blocked_ips.clear()
    fw.simulated_acl_blocked_ips.update(saved[2])
    fw.audit_log_path = saved[3]


def test_api_block_in_dry_run_is_simulated(admin, os_calls):
    r = admin.client.post("/api/execute_real_block", json={"ip": "203.0.113.20"}, headers=admin.headers)
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "SIMULATED_BLOCK_RECORDED" and body["dry_run"] is True
    assert os_calls == []
    assert read_audit(admin.audit)[0]["actor"].startswith("admin@")


def test_api_enforce_requires_confirmation(admin, os_calls):
    admin.fw.mode = "enforce"
    r = admin.client.post("/api/execute_real_block", json={"ip": "203.0.113.21"}, headers=admin.headers)
    assert r.status_code == 428 and r.get_json()["status"] == "CONFIRMATION_REQUIRED"
    assert os_calls == []
    r = admin.client.post("/api/execute_real_block", json={"ip": "203.0.113.21", "confirm": True},
                          headers=admin.headers)
    assert r.status_code == 200 and r.get_json()["status"] == "REAL_FIREWALL_RULE_APPLIED"
    assert os_calls  # a real command ran only after explicit confirmation in enforce mode


def test_api_mode_change_requires_admin_and_explicit_confirm(admin):
    r = admin.client.post("/api/firewall/mode", json={"mode": "enforce", "confirm": "ENFORCE"})
    assert r.status_code == 403
    r = admin.client.post("/api/firewall/mode", json={"mode": "enforce"}, headers=admin.headers)
    assert r.status_code == 428 and admin.fw.mode == "dry-run"
    r = admin.client.post("/api/firewall/mode", json={"mode": "bogus"}, headers=admin.headers)
    assert r.status_code == 400
    r = admin.client.post("/api/firewall/mode", json={"mode": "enforce", "confirm": "ENFORCE"},
                          headers=admin.headers)
    assert r.status_code == 200 and admin.fw.mode == "enforce"
    assert read_audit(admin.audit)[-1]["action"] == "MODE_CHANGE"
    r = admin.client.post("/api/firewall/mode", json={"mode": "dry-run"}, headers=admin.headers)
    assert r.status_code == 200 and admin.fw.mode == "dry-run"


def test_api_mode_get_reports_state(admin):
    r = admin.client.get("/api/firewall/mode")
    assert r.status_code == 200 and r.get_json()["mode"] == "dry-run"


def test_api_cannot_leave_enforce_with_active_rules(admin, os_calls):
    admin.fw.mode = "enforce"
    admin.client.post("/api/execute_real_block", json={"ip": "203.0.113.22", "confirm": True}, headers=admin.headers)
    r = admin.client.post("/api/firewall/mode", json={"mode": "dry-run"}, headers=admin.headers)
    assert r.status_code == 409 and admin.fw.mode == "enforce"
