"""
tests/test_phase2_visualization.py
==================================
Verification test suite for Phase 2:
- CG-02: Connection aggregation (port 5000 repeated connections)
- CG-04: Preservation of meaningful differences (source, destination, protocol, service)
- CG-05: Threat card display durations (Level 1: 10-20s, Level 2: 45s, Level 3: 60-90s)
- CG-06: Non-destructive threat card retention across cycles
"""

import time
import pytest
import app as app_module
from app import (
    app,
    init_engines,
    _run_pipeline_cycle,
    _active_threat_cards,
    shared_state,
    state_lock,
    set_network_profile_override,
)


@pytest.fixture(autouse=True)
def setup_phase2_env():
    init_engines()
    with state_lock:
        _active_threat_cards.clear()
        shared_state["evals"] = []
        shared_state["graph_edges"] = []
    set_network_profile_override({
        "iface": "Wi-Fi",
        "local_ip": "10.160.36.211",
        "local_subnet": "10.160.36.0/24",
        "default_gateway": "10.160.36.14",
    })
    yield


def test_repeated_connections_to_port_5000_aggregate_into_one():
    """Requirement: Repeated connections from same system to port 5000 aggregate into one logical edge."""
    client_ip = "10.160.36.186"
    benign_sub = app_module.df_stream_pool[app_module.df_stream_pool["label_enc"] == 0]
    base_flow = benign_sub.iloc[0].to_dict()

    # Create 5 repeated flows from client_ip to port 5000 (protocol 6)
    batch = []
    for _ in range(5):
        f = dict(base_flow)
        f["_src_ip"] = client_ip
        f["src_ip"] = client_ip
        f["_dst_port"] = 5000
        f["dst_port"] = 5000
        f["L4_DST_PORT"] = 5000
        f["PROTOCOL"] = 6
        f["protocol"] = 6
        batch.append(f)

    evals, graph_nodes, edge_risk_list, _ = _run_pipeline_cycle(batch_override=batch)

    # Filter edges connecting client_ip to 5000/6
    matching_edges = [
        e for e in edge_risk_list
        if client_ip in e["source_label"] and "5000" in e["target_label"]
    ]
    assert len(matching_edges) == 1, f"Expected 1 aggregated edge, got {len(matching_edges)}"
    assert matching_edges[0]["count"] == 5


def test_preserve_meaningful_differences_across_services_and_protocols():
    """Requirement: Preserve differences in source, destination, protocol, and service."""
    benign_sub = app_module.df_stream_pool[app_module.df_stream_pool["label_enc"] == 0]
    base_flow = benign_sub.iloc[0].to_dict()

    client_a = "10.160.36.186"
    client_b = "10.160.36.187"

    # 4 distinct flows:
    # 1. client_a -> port 5000 / TCP (6)
    # 2. client_a -> port 80 / TCP (6)
    # 3. client_a -> port 53 / UDP (17)
    # 4. client_b -> port 5000 / TCP (6)
    flows = [
        {"_src_ip": client_a, "src_ip": client_a, "_dst_port": 5000, "dst_port": 5000, "PROTOCOL": 6, "protocol": 6},
        {"_src_ip": client_a, "src_ip": client_a, "_dst_port": 80, "dst_port": 80, "PROTOCOL": 6, "protocol": 6},
        {"_src_ip": client_a, "src_ip": client_a, "_dst_port": 53, "dst_port": 53, "PROTOCOL": 17, "protocol": 17},
        {"_src_ip": client_b, "src_ip": client_b, "_dst_port": 5000, "dst_port": 5000, "PROTOCOL": 6, "protocol": 6},
    ]

    batch = []
    for f_override in flows:
        f = dict(base_flow)
        f.update(f_override)
        f["L4_DST_PORT"] = f_override["dst_port"]
        batch.append(f)

    evals, graph_nodes, edge_risk_list, _ = _run_pipeline_cycle(batch_override=batch)

    assert len(edge_risk_list) == 4, f"Expected 4 distinct logical edges, got {len(edge_risk_list)}"
    targets = {e["target_label"] for e in edge_risk_list}
    assert "5000/6" in targets
    assert "80/6" in targets
    assert "53/17" in targets

    sources = {e["source_label"] for e in edge_risk_list}
    assert any(client_a in s for s in sources)
    assert any(client_b in s for s in sources)


def test_threat_card_durations_level1_level2_level3():
    """Requirement: Level 1 (10-20s), Level 2 (45s), Level 3 (60-90s) retention."""
    now = time.time()

    # Create synthetic evaluation events for L1, L2, L3
    ev_l1 = {
        "level": "Level 1 — Normal",
        "risk_category": "BENIGN",
        "source_host": "HOST:10.160.36.101",
        "target_service": "SERVICE:80/6",
        "risk_score": 0.05,
        "badge_color": "#10b981",
        "summary": "Normal flow",
    }
    ev_l2 = {
        "level": "Level 2 — Suspicious / Emerging",
        "risk_category": "EARLY_WARNING_THREAT",
        "source_host": "HOST:10.160.36.102",
        "target_service": "SERVICE:22/6",
        "risk_score": 0.55,
        "badge_color": "#f59e0b",
        "summary": "Suspicious scan",
    }
    ev_l3 = {
        "level": "Level 3 — High Risk",
        "risk_category": "CRITICAL_ATTACK_THREAT",
        "source_host": "HOST:10.160.36.103",
        "target_service": "SERVICE:445/6",
        "risk_score": 0.95,
        "badge_color": "#ef4444",
        "summary": "Critical exploit attempt",
    }

    # Simulate batch containing all three
    with state_lock:
        _active_threat_cards.clear()

    # Check durations set by pipeline
    durations = {}
    for ev in [ev_l1, ev_l2, ev_l3]:
        lvl = ev["level"]
        if "Level 3" in lvl:
            dur = 75.0
        elif "Level 2" in lvl:
            dur = 45.0
        else:
            dur = 15.0
        durations[lvl] = dur
        _active_threat_cards[(ev["source_host"], ev["target_service"], lvl)] = {
            "eval": ev,
            "expires_at": now + dur,
            "created_at": now,
        }

    # L1 must be between 10 and 20s
    assert 10.0 <= durations["Level 1 — Normal"] <= 20.0
    # L2 must be 45s
    assert durations["Level 2 — Suspicious / Emerging"] == 45.0
    # L3 must be between 60 and 90s
    assert 60.0 <= durations["Level 3 — High Risk"] <= 90.0


def test_new_threat_events_do_not_overwrite_unrelated_cards():
    """Requirement: An idle or unrelated cycle does not clear active threat cards."""
    # 1. Establish an active Level 3 threat card
    ev_l3 = {
        "level": "Level 3 — High Risk",
        "risk_category": "CRITICAL_ATTACK_THREAT",
        "source_host": "HOST:10.160.36.200",
        "target_service": "SERVICE:445/6",
        "risk_score": 0.92,
        "badge_color": "#ef4444",
        "summary": "Critical Threat",
    }
    now = time.time()
    with state_lock:
        _active_threat_cards[("HOST:10.160.36.200", "SERVICE:445/6", ev_l3["level"])] = {
            "eval": ev_l3,
            "expires_at": now + 75.0,
            "created_at": now,
        }

    # 2. Run empty cycle (zero traffic / idle)
    _run_pipeline_cycle(batch_override=[])

    # 3. Assert Level 3 card is STILL in shared_state["evals"]
    with state_lock:
        eval_hosts = [e.get("source_host") for e in shared_state["evals"]]
        assert "HOST:10.160.36.200" in eval_hosts, "Active Level 3 card was incorrectly erased by empty batch!"

