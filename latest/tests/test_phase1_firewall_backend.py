"""
tests/test_phase1_firewall_backend.py
=====================================
Verification test suite for Phase 1:
- CG-07, CG-08, CG-09, CG-10
- Level 3 blocking workflow & verification
- Missing /api/unblock_ip endpoint
- Admin authorization & X-Forwarded-For anti-spoofing
- Single LiveTrafficEngine initialization (no duplicates)
"""

import pytest
import ipaddress
from app import (
    app,
    init_engines,
    fw_executor,
    traffic_engine,
    is_admin_request,
    get_admin_token,
    set_network_profile_override,
    _authorized_monitored_devices,
    state_lock,
)
import live_capture


@pytest.fixture(autouse=True)
def setup_phase1_env():
    """Ensure engines are initialized and reset state for tests."""
    init_engines()
    with state_lock:
        _authorized_monitored_devices.clear()
    set_network_profile_override({
        "iface": "Wi-Fi",
        "local_ip": "10.160.36.211",
        "local_subnet": "10.160.36.0/24",
        "default_gateway": "10.160.36.14",
    })
    yield


def test_no_duplicate_live_traffic_engine_in_live_capture():
    """Requirement: live_capture does not instantiate a duplicate LiveTrafficEngine."""
    assert not hasattr(live_capture, "_engine") or live_capture._engine is None
    # Verify LiveTrafficEngine is not imported or stored in live_capture module
    assert "LiveTrafficEngine" not in live_capture.__dict__


def test_admin_authorization_prevents_unauthorized_execute_real_block():
    """Requirement: Remote client receives 403 when calling execute_real_block without admin token, and GET is blocked (405)."""
    client = app.test_client()
    remote_ip = "10.160.36.186"

    # GET is rejected to prevent CSRF via <img src> or <a> tag
    res_get = client.get("/api/execute_real_block?ip=10.0.0.99", environ_base={"REMOTE_ADDR": remote_ip})
    assert res_get.status_code == 405

    # POST without admin token receives 403 Forbidden
    res = client.post("/api/execute_real_block", json={"ip": "10.0.0.99"}, environ_base={"REMOTE_ADDR": remote_ip})
    assert res.status_code == 403
    data = res.get_json()
    assert data["status"] == "ERROR"
    assert "Forbidden" in data["message"]


def test_admin_authorization_prevents_unauthorized_unblock():
    """Requirement: Remote client receives 403 when calling unblock_ip without admin token, and GET is blocked (405)."""
    client = app.test_client()
    remote_ip = "10.160.36.186"

    # GET is rejected to prevent CSRF
    res_get = client.get("/api/unblock_ip?ip=10.0.0.99", environ_base={"REMOTE_ADDR": remote_ip})
    assert res_get.status_code == 405

    # POST without admin token receives 403 Forbidden
    res = client.post("/api/unblock_ip", json={"ip": "10.0.0.99"}, environ_base={"REMOTE_ADDR": remote_ip})
    assert res.status_code == 403
    data = res.get_json()
    assert data["status"] == "ERROR"
    assert "Forbidden" in data["message"]


def test_anti_spoofing_arbitrary_x_forwarded_for_rejected():
    """Requirement: Remote client sending X-Forwarded-For: 127.0.0.1 cannot gain admin."""
    client = app.test_client()
    remote_ip = "10.160.36.186"

    # Spoof attempt: remote client claims loopback in X-Forwarded-For on POST
    res = client.post(
        "/api/execute_real_block",
        json={"ip": "10.0.0.99"},
        headers={"X-Forwarded-For": "127.0.0.1"},
        environ_base={"REMOTE_ADDR": remote_ip},
    )
    assert res.status_code == 403

    res_unblock = client.post(
        "/api/unblock_ip",
        json={"ip": "10.0.0.99"},
        headers={"X-Forwarded-For": "127.0.0.1"},
        environ_base={"REMOTE_ADDR": remote_ip},
    )
    assert res_unblock.status_code == 403


def test_admin_execute_block_and_verify_simulation_mode():
    """Requirement: Admin can block IP in simulation mode and verify status with valid Admin Token."""
    client = app.test_client()
    target_ip = "192.168.1.188"
    token = get_admin_token()

    # Ensure unblocked first
    from app import fw_executor
    fw_executor.execute_unblock(target_ip)

    # Admin executes block via POST with X-Admin-Token
    res = client.post(
        "/api/execute_real_block",
        json={"ip": target_ip},
        headers={"X-Admin-Token": token},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "verification" in data
    assert data["verification"]["is_blocked"] is True
    assert data["verification"]["simulated_memory_blocked"] is True

    # Check dedicated verify endpoint
    v_res = client.get(f"/api/verify_blocked?ip={target_ip}")
    assert v_res.status_code == 200
    v_data = v_res.get_json()
    assert v_data["is_blocked"] is True
    assert v_data["status_description"] == "BLOCKED & VERIFIED"


def test_admin_unblock_ip_endpoint_restores_allowed_status():
    """Requirement: /api/unblock_ip unblocks previously blocked IP via POST with Admin Token."""
    client = app.test_client()
    target_ip = "192.168.1.189"
    token = get_admin_token()

    # Block IP first
    from app import fw_executor
    fw_executor.execute_block(target_ip)
    assert fw_executor.verify_blocked(target_ip)["is_blocked"] is True

    # Call /api/unblock_ip as Admin via POST
    res = client.post(
        "/api/unblock_ip",
        json={"ip": target_ip},
        headers={"X-Admin-Token": token},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "UNBLOCKED_SUCCESS"
    assert data["verification"]["is_blocked"] is False
    assert data["verification"]["status_description"] == "UNBLOCKED / ALLOWED"

    # Verify directly
    assert fw_executor.verify_blocked(target_ip)["is_blocked"] is False


def test_invalid_ip_handling():
    """Requirement: Malformed IP addresses return 400 Bad Request."""
    client = app.test_client()
    token = get_admin_token()

    res_block = client.post(
        "/api/execute_real_block",
        json={"ip": "not-an-ip"},
        headers={"X-Admin-Token": token},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert res_block.status_code == 400
    assert "Invalid IP" in res_block.get_json()["message"]

    res_unblock = client.post(
        "/api/unblock_ip",
        json={"ip": "invalid_addr"},
        headers={"X-Admin-Token": token},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert res_unblock.status_code == 400
    assert "Invalid IP" in res_unblock.get_json()["message"]


def test_protected_whitelist_and_ipv6_handling():
    """Requirement: Gateway and Server IPs cannot be blocked; IPv6 parsed accurately."""
    client = app.test_client()
    token = get_admin_token()

    # Attempt to block Gateway IP (10.160.36.14)
    res_gw = client.post(
        "/api/execute_real_block",
        json={"ip": "10.160.36.14"},
        headers={"X-Admin-Token": token},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert res_gw.status_code == 200
    data_gw = res_gw.get_json()
    assert data_gw["status"] == "ERROR"
    assert data_gw["execution_mode"] == "PROTECTED_WHITELIST"

    # Test IPv6 address support
    res_ipv6 = client.post(
        "/api/execute_real_block",
        json={"ip": "2001:db8::cafe"},
        headers={"X-Admin-Token": token},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert res_ipv6.status_code == 200
    assert res_ipv6.get_json()["verification"]["is_blocked"] is True

