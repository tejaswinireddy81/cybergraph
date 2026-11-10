"""
tests/test_final_acceptance_scenario.py
=========================================
Automated test script executing the exact 6-step Final Acceptance Scenario (Requirement 16).
"""

import pytest
import time
import app as app_module
from app import (
    resolve_node_type,
    process_client_request,
    approve_device_request,
    get_pending_requests,
    get_authorized_devices,
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


@pytest.mark.requires_models
def test_final_acceptance_scenario():
    """Run the 6-step Final Acceptance Test Scenario."""
    init_engines()
    with state_lock:
        _authorized_monitored_devices.clear()
        _pending_device_requests.clear()
        _discovered_lan_hosts.clear()
        _default_gateways.clear()

    set_network_profile_override({
        "iface": "wlan0",
        "local_ip": "10.160.36.211",
        "local_subnet": "10.160.36.0/24",
        "default_gateway": "10.160.36.14",
    })

    server_ip = "10.160.36.211"
    friend_ip = "10.160.36.186"
    gateway_ip = "10.160.36.14"

    # -------------------------------------------------------------------
    # STEP 1: Only server is running.
    # Expected: HOST: 10.160.36.211
    # -------------------------------------------------------------------
    print("\n[ACCEPTANCE TEST] Step 1: Server running...")
    evals, graph_nodes, edges, _ = _run_pipeline_cycle(batch_override=[])
    host_labels = [n["label"] for n in graph_nodes if n["type"] == "HOST"]

    assert server_ip in host_labels
    assert resolve_node_type(server_ip) == "HOST"
    assert resolve_node_type(friend_ip) == "EXTERNAL"
    assert resolve_node_type(gateway_ip) == "GATEWAY"

    # -------------------------------------------------------------------
    # STEP 2: Friend opens http://10.160.36.211:5000
    # Expected: PENDING ACCESS REQUEST: 10.160.36.186 (Must NOT immediately become HOST)
    # -------------------------------------------------------------------
    print("[ACCEPTANCE TEST] Step 2: Friend requests web access...")
    pending_req = process_client_request(client_ip=friend_ip, device_token="friend-laptop-token")
    assert pending_req is not None
    assert pending_req["status"] == "PENDING"
    assert pending_req["client_ip"] == friend_ip

    # Must NOT immediately become HOST (classified as PENDING / Suspicious)
    assert resolve_node_type(friend_ip) == "PENDING"
    assert resolve_node_type(friend_ip) != "HOST"

    evals, graph_nodes, edges, _ = _run_pipeline_cycle(batch_override=[])
    host_labels = [n["label"] for n in graph_nodes if n["type"] == "HOST"]
    assert friend_ip not in host_labels

    # -------------------------------------------------------------------
    # STEP 3: Administrator clicks APPROVE.
    # Expected: HOST: 10.160.36.211, HOST: 10.160.36.186
    # -------------------------------------------------------------------
    print("[ACCEPTANCE TEST] Step 3: Admin approves friend laptop...")
    approved = approve_device_request(pending_req["request_id"])
    assert approved is not None
    assert approved["status"] == "AUTHORIZED"

    assert resolve_node_type(server_ip) == "HOST"
    assert resolve_node_type(friend_ip) == "HOST"

    evals, graph_nodes, edges, _ = _run_pipeline_cycle(batch_override=[])
    host_labels = [n["label"] for n in graph_nodes if n["type"] == "HOST"]
    assert server_ip in host_labels
    assert friend_ip in host_labels

    # -------------------------------------------------------------------
    # STEP 4: Friend laptop sends no traffic for 30+ seconds (simulated empty batches).
    # Expected: Both HOST nodes remain visible; 10.160.36.186 risk is Normal/neutral.
    # -------------------------------------------------------------------
    print("[ACCEPTANCE TEST] Step 4: 30+ seconds zero traffic persistence...")
    for _ in range(10):
        evals, graph_nodes, edges, _ = _run_pipeline_cycle(batch_override=[])
        host_nodes = {n["label"]: n for n in graph_nodes if n["type"] == "HOST"}
        assert server_ip in host_nodes
        assert friend_ip in host_nodes
        assert host_nodes[friend_ip]["threat_level"] == "NORMAL"
        assert host_nodes[friend_ip]["max_risk"] == 0.0

    # -------------------------------------------------------------------
    # STEP 5: Other random external/private endpoints appear in traffic.
    # Expected: They are EXTERNAL. They must NOT create additional persistent HOST nodes.
    # -------------------------------------------------------------------
    print("[ACCEPTANCE TEST] Step 5: Random traffic from unapproved IPs...")
    benign_sub = app_module.df_stream_pool[app_module.df_stream_pool["label_enc"] == 0]

    f1 = benign_sub.iloc[0].to_dict()
    f1["_src_ip"] = "13.107.7.93"
    f1["src_ip"] = "13.107.7.93"
    f1["_dst_ip"] = server_ip
    f1["dst_ip"] = server_ip

    f2 = benign_sub.iloc[1].to_dict()
    f2["_src_ip"] = "10.48.85.10"
    f2["src_ip"] = "10.48.85.10"
    f2["_dst_ip"] = friend_ip
    f2["dst_ip"] = friend_ip

    f3 = benign_sub.iloc[2].to_dict()
    f3["_src_ip"] = "192.168.0.7"
    f3["src_ip"] = "192.168.0.7"
    f3["_dst_ip"] = server_ip
    f3["dst_ip"] = server_ip

    evals, graph_nodes, edges, _ = _run_pipeline_cycle(batch_override=[f1, f2, f3])

    # Check node types for random endpoints
    assert resolve_node_type("13.107.7.93") == "EXTERNAL"
    assert resolve_node_type("10.48.85.10") == "EXTERNAL"
    assert resolve_node_type("192.168.0.7") == "EXTERNAL"

    # Verify HOST nodes in graph are STILL ONLY authorized hosts (server_ip and friend_ip)
    host_labels = set(n["label"] for n in graph_nodes if n["type"] == "HOST")
    assert host_labels == {server_ip, friend_ip}

    # -------------------------------------------------------------------
    # STEP 6: Attack traffic generated from authorized friend laptop.
    # Expected: 10.160.36.186 remains HOST and its risk/edges can turn Suspicious/Critical.
    # Attack detection behavior remains active and accurate.
    # -------------------------------------------------------------------
    print("[ACCEPTANCE TEST] Step 6: Attack traffic from authorized friend laptop...")
    attack_sub = app_module.df_stream_pool[app_module.df_stream_pool["label_enc"] == 1]
    f_attack = attack_sub.iloc[0].to_dict()
    f_attack["_src_ip"] = friend_ip
    f_attack["src_ip"] = friend_ip
    f_attack["_dst_ip"] = server_ip
    f_attack["dst_ip"] = server_ip

    evals, graph_nodes, edges, _ = _run_pipeline_cycle(batch_override=[f_attack])

    # 10.160.36.186 MUST STILL BE A HOST NODE
    assert resolve_node_type(friend_ip) == "HOST"

    host_nodes = {n["label"]: n for n in graph_nodes if n["type"] == "HOST"}
    assert friend_ip in host_nodes
    assert host_nodes[friend_ip]["threat_level"] in ("CRITICAL", "SUSPICIOUS")
    assert host_nodes[friend_ip]["max_risk"] > 0.4

    print("\n[ACCEPTANCE TEST SUCCESS] All 6 steps completed cleanly and verified!")
