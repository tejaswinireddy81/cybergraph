"""
tests/test_phase4_reliability_and_regression.py
===============================================
Verification test suite for Phase 4:
- CG-11: Polling mechanism & overlapping request protection
- CG-12: Avoidance of unnecessary repeated device-access calls
- CG-13: Safe dynamic HTML rendering & XSS prevention
- Comprehensive API reliability and error handling
"""

import time
import pytest
from app import (
    app,
    init_engines,
    state_lock,
    set_network_profile_override,
)


@pytest.fixture(autouse=True)
def setup_phase4_env():
    init_engines()
    set_network_profile_override({
        "iface": "Wi-Fi",
        "local_ip": "10.160.36.211",
        "local_subnet": "10.160.36.0/24",
        "default_gateway": "10.160.36.14",
    })
    yield


def test_client_script_contains_non_overlapping_polling_and_device_gate():
    """Requirement: Frontend implements non-overlapping polling and state_version gating for device requests."""
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # 1. Verify isPolling guard is implemented
    assert "isPolling" in html

    # 2. Verify recursive scheduleNextPoll / setTimeout replaces blind setInterval
    assert "scheduleNextPoll" in html
    assert "setInterval(fetchStreamStep, 750)" not in html

    # 3. Verify device access section is gated by state_version
    assert "lastDeviceStateVersion" in html
    assert "data.state_version !== lastDeviceStateVersion" in html


def test_html_and_client_contain_xss_protection_helpers():
    """Requirement: escapeHtml and sanitizeIp helpers are implemented and used for dynamic rendering."""
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Verify escapeHtml function is defined
    assert "function escapeHtml(str)" in html
    assert "function sanitizeIp(str)" in html
    assert "escapeHtml(ev.level" in html
    assert "escapeHtml(ev.source_host" in html
    assert "escapeHtml(ev.summary" in html


def test_api_error_handling_malformed_inputs():
    """Requirement: All API endpoints handle missing or malformed inputs with 400 Bad Request, never 500."""
    from app import get_admin_token
    token = get_admin_token()
    client = app.test_client()

    # GET to state-changing routes returns 405 Method Not Allowed
    assert client.get("/api/execute_real_block").status_code == 405
    assert client.get("/api/unblock_ip").status_code == 405

    # Missing params on POST with Admin Token
    res1 = client.post("/api/execute_real_block", json={}, headers={"X-Admin-Token": token}, environ_base={"REMOTE_ADDR": "127.0.0.1"})
    assert res1.status_code == 400
    assert "Missing IP" in res1.get_json()["message"]

    res2 = client.post("/api/unblock_ip", json={}, headers={"X-Admin-Token": token}, environ_base={"REMOTE_ADDR": "127.0.0.1"})
    assert res2.status_code == 400
    assert "Missing IP" in res2.get_json()["message"]

    res3 = client.get("/api/verify_blocked")
    assert res3.status_code == 400
    assert "Missing IP" in res3.get_json()["message"]

    # Invalid IPs on POST with Admin Token
    res4 = client.post("/api/execute_real_block", json={"ip": "not_an_ip"}, headers={"X-Admin-Token": token}, environ_base={"REMOTE_ADDR": "127.0.0.1"})
    assert res4.status_code == 400
    assert "Invalid IP" in res4.get_json()["message"]

    res5 = client.post("/api/unblock_ip", json={"ip": "999.999.999.999"}, headers={"X-Admin-Token": token}, environ_base={"REMOTE_ADDR": "127.0.0.1"})
    assert res5.status_code == 400
    assert "Invalid IP" in res5.get_json()["message"]

    # Missing parameters for device actions
    res6 = client.post("/api/approve_device", json={}, headers={"X-Admin-Token": token}, environ_base={"REMOTE_ADDR": "127.0.0.1"})
    assert res6.status_code == 400

    res7 = client.post("/api/revoke_device", json={}, headers={"X-Admin-Token": token}, environ_base={"REMOTE_ADDR": "127.0.0.1"})
    assert res7.status_code == 400


def test_stream_step_endpoint_reliability_under_rapid_polling():
    """Requirement: /api/stream_step responds reliably without deadlock or errors under rapid requests."""
    client = app.test_client()

    for _ in range(10):
        res = client.get("/api/stream_step")
        assert res.status_code == 200
        data = res.get_json()
        assert "stats" in data
        assert "graph_nodes" in data
        assert "graph_edges" in data
        assert "state_version" in data

