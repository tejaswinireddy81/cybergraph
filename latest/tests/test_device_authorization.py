"""
tests/test_device_authorization.py
===================================
Comprehensive unit test suite for CyberGraph Monitored-Device Authorization Architecture.
Verifies all core requirements (Tests A through O).
"""

import pytest
import time
import copy
import app as app_module
from app import (
    app,
    resolve_node_type,
    format_endpoint_label,
    is_local_lan_host,
    set_server_local_ip,
    get_server_local_ip,
    process_client_request,
    approve_device_request,
    reject_device_request,
    revoke_device_authorization,
    get_authorized_devices,
    get_pending_requests,
    sync_network_state,
    set_network_profile_override,
    _run_pipeline_cycle,
    _authorized_monitored_devices,
    _pending_device_requests,
    _discovered_lan_hosts,
    _default_gateways,
    init_engines,
    state_lock,
)


@pytest.fixture(autouse=True)
def reset_authorization_state():
    """Reset global state before each test."""
    init_engines()
    with state_lock:
        _authorized_monitored_devices.clear()
        _pending_device_requests.clear()
        _discovered_lan_hosts.clear()
        _default_gateways.clear()

    # Set mock server profile for the active network
    set_network_profile_override({
        "iface": "Wi-Fi",
        "local_ip": "10.160.36.211",
        "local_subnet": "10.160.36.0/24",
        "default_gateway": "10.160.36.14",
    })
    yield


def test_a_only_active_monitoring_ip_is_server():
    """Requirement A: Only active monitoring IP is CyberGraph Server."""
    srv_ip = get_server_local_ip()
    assert srv_ip == "10.160.36.211"
    assert resolve_node_type(srv_ip) == "HOST"

    auth_devices = get_authorized_devices()
    server_entries = [d for d in auth_devices if d.get("is_server")]
    assert len(server_entries) == 1
    assert server_entries[0]["client_ip"] == "10.160.36.211"


def test_b_secondary_virtual_ip_cannot_become_server():
    """Requirement B: 10.41.92.156 cannot become CyberGraph Server when it belongs to a secondary/virtual interface."""
    secondary_ip = "10.41.92.156"

    # Simulate secondary IP visiting or connecting
    req = process_client_request(client_ip=secondary_ip, device_token="token-virtual-adapter")
    assert req is not None
    assert req["status"] == "PENDING"
    assert req.get("is_server") is False

    # Verify it is NOT classified as HOST
    assert resolve_node_type(secondary_ip) != "HOST"

    # Verify it never entered authorized devices as server
    auth_devices = get_authorized_devices()
    for dev in auth_devices:
        if dev.get("client_ip") == secondary_ip:
            assert dev.get("is_server") is not True


def test_c_exactly_one_is_server_true_entry_exists():
    """Requirement C: Exactly one is_server=True entry exists."""
    auth_devices = get_authorized_devices()
    server_entries = [d for d in auth_devices if d.get("is_server")]
    assert len(server_entries) == 1
    assert server_entries[0]["client_ip"] == get_server_local_ip()

    # Multiple client requests from loopback / server must not duplicate server entries
    process_client_request("127.0.0.1")
    process_client_request("10.160.36.211")

    auth_devices_after = get_authorized_devices()
    server_entries_after = [d for d in auth_devices_after if d.get("is_server")]
    assert len(server_entries_after) == 1
    assert server_entries_after[0]["client_ip"] == "10.160.36.211"


def test_d_pending_device_starts_pending():
    """Requirement D: 10.160.36.186 starts PENDING."""
    client_ip = "10.160.36.186"
    req = process_client_request(client_ip=client_ip, device_token="token-186")
    assert req is not None
    assert req["status"] == "PENDING"
    assert req["client_ip"] == client_ip
    assert req.get("is_server") is False

    # Classification is PENDING
    assert resolve_node_type(client_ip) == "PENDING"
    assert format_endpoint_label(client_ip) == "PENDING:10.160.36.186"


def test_e_pending_device_is_not_authorized_host():
    """Requirement E: Pending device is NOT an authorized HOST."""
    client_ip = "10.160.36.186"
    process_client_request(client_ip=client_ip, device_token="token-186")

    assert resolve_node_type(client_ip) != "HOST"
    assert is_local_lan_host(client_ip) is False

    auth_ips = [d["client_ip"] for d in get_authorized_devices()]
    assert client_ip not in auth_ips


def test_f_pending_device_graph_node_rendered_as_pending_suspicious():
    """Requirement F: Pending device is visually/classified as SUSPICIOUS/PENDING."""
    client_ip = "10.160.36.186"
    process_client_request(client_ip=client_ip, device_token="token-186")

    # Simulate flow from pending client
    benign_sub = app_module.df_stream_pool[app_module.df_stream_pool["label_enc"] == 0]
    flow = benign_sub.iloc[0].to_dict()
    flow["_src_ip"] = client_ip
    flow["src_ip"] = client_ip
    flow["_dst_ip"] = "10.160.36.211"
    flow["dst_ip"] = "10.160.36.211"

    evals, graph_nodes, edges, _ = _run_pipeline_cycle(batch_override=[flow])

    # Find pending node in graph
    pending_nodes = [n for n in graph_nodes if client_ip in n["label"]]
    assert len(pending_nodes) == 1
    node = pending_nodes[0]
    assert node["type"] == "PENDING"
    assert node["color"] == "suspicious"
    assert node["threat_level"] == "SUSPICIOUS"
    assert node["label"] == f"PENDING:{client_ip}"


def test_g_only_admin_server_can_approve():
    """Requirement G: Only admin/server can approve."""
    client = app.test_client()
    friend_ip = "10.160.36.186"
    req = process_client_request(client_ip=friend_ip, device_token="token-admin-test")

    # Admin with valid admin token can approve
    from app import get_admin_token
    res = client.post(
        "/api/approve_device",
        json={"request_id": req["request_id"]},
        headers={"X-Admin-Token": get_admin_token()},
        environ_base={"REMOTE_ADDR": "127.0.0.1"}
    )
    assert res.status_code == 200
    assert resolve_node_type(friend_ip) == "HOST"


def test_h_remote_client_receives_403_from_approve():
    """Requirement H: Remote client receives 403 from approve endpoint."""
    client = app.test_client()
    friend_ip = "10.160.36.186"
    req = process_client_request(client_ip=friend_ip, device_token="token-remote-403")

    # Remote client tries to approve -> 403 Forbidden
    res = client.post(
        "/api/approve_device",
        json={"request_id": req["request_id"]},
        environ_base={"REMOTE_ADDR": friend_ip}
    )
    assert res.status_code == 403
    assert resolve_node_type(friend_ip) == "PENDING"  # Remains unapproved


def test_i_after_approval_becomes_host():
    """Requirement I: After approval, 10.160.36.186 becomes HOST."""
    friend_ip = "10.160.36.186"
    req = process_client_request(client_ip=friend_ip, device_token="token-host-test")

    approved = approve_device_request(req["request_id"])
    assert approved is not None
    assert approved["status"] == "AUTHORIZED"

    assert resolve_node_type(friend_ip) == "HOST"
    assert format_endpoint_label(friend_ip) == "HOST:10.160.36.186"
    assert is_local_lan_host(friend_ip) is True


def test_j_approved_host_persists_with_zero_traffic():
    """Requirement J: Approved host persists with zero traffic."""
    friend_ip = "10.160.36.186"
    approve_device_request(friend_ip)

    # Run multiple cycles with zero traffic (empty batch)
    for _ in range(5):
        evals, graph_nodes, edges, _ = _run_pipeline_cycle(batch_override=[])
        host_labels = [n["label"] for n in graph_nodes if n["type"] == "HOST"]
        assert friend_ip in host_labels
        assert resolve_node_type(friend_ip) == "HOST"


def test_k_remote_client_cannot_revoke():
    """Requirement K: Remote client cannot revoke."""
    client = app.test_client()
    friend_ip = "10.160.36.186"
    approve_device_request(friend_ip)
    assert resolve_node_type(friend_ip) == "HOST"

    # Remote client attempts to revoke -> 403 Forbidden
    res = client.post(
        "/api/revoke_device",
        json={"ip": friend_ip},
        environ_base={"REMOTE_ADDR": friend_ip}
    )
    assert res.status_code == 403
    assert resolve_node_type(friend_ip) == "HOST"  # Authorization unchanged


def test_l_reject_button_does_not_exist_in_rendered_dashboard():
    """Requirement L: Reject button does not exist in rendered dashboard."""
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Assert REJECT button is NOT present anywhere in HTML or JavaScript
    assert "REJECT</button>" not in html
    assert 'onclick="rejectDevice' not in html
    assert "rejectDevice(" not in html


def test_m_unknown_private_ip_never_becomes_host():
    """Requirement M: Unknown private IP never becomes HOST."""
    assert resolve_node_type("10.48.85.10") == "EXTERNAL"
    assert resolve_node_type("192.168.1.99") == "EXTERNAL"
    assert resolve_node_type("172.16.5.5") == "EXTERNAL"


def test_n_network_switch_clears_old_authorizations():
    """Requirement N: Network switch clears old authorizations."""
    friend_ip = "10.160.36.186"
    approve_device_request(friend_ip)
    assert resolve_node_type(friend_ip) == "HOST"

    # Switch network to 192.168.0.0/24
    set_network_profile_override({
        "iface": "eth0",
        "local_ip": "192.168.0.50",
        "local_subnet": "192.168.0.0/24",
        "default_gateway": "192.168.0.1",
    })

    # Friend IP from old network MUST NO LONGER BE HOST
    assert resolve_node_type(friend_ip) == "EXTERNAL"

    # New server IP MUST be HOST automatically
    assert resolve_node_type("192.168.0.50") == "HOST"
    assert resolve_node_type("192.168.0.1") == "GATEWAY"

    # Exactly one server device exists
    server_entries = [d for d in get_authorized_devices() if d.get("is_server")]
    assert len(server_entries) == 1
    assert server_entries[0]["client_ip"] == "192.168.0.50"


def test_o_no_hard_coded_ip_addresses():
    """Requirement O: Dynamic authorization works without hard-coded IPs."""
    dynamic_server = "172.20.10.2"
    dynamic_friend = "172.20.10.5"
    dynamic_gw = "172.20.10.1"

    set_network_profile_override({
        "iface": "eth_dyn",
        "local_ip": dynamic_server,
        "local_subnet": "172.20.10.0/24",
        "default_gateway": dynamic_gw,
    })

    assert resolve_node_type(dynamic_server) == "HOST"
    assert resolve_node_type(dynamic_gw) == "GATEWAY"
    assert resolve_node_type(dynamic_friend) == "EXTERNAL"

    # Request & approve
    req = process_client_request(dynamic_friend)
    assert req["status"] == "PENDING"
    assert resolve_node_type(dynamic_friend) == "PENDING"

    approve_device_request(dynamic_friend)
    assert resolve_node_type(dynamic_friend) == "HOST"
