"""
tests/test_attack_journey_engine.py
====================================
Unit Tests for CyberGraph Attack Journey Intelligence Engine
"""

import pytest
from modules.attack_journey_engine import AttackJourneyEngine
from modules.bari_engine import BARIEngine


def test_journey_reconstruction_and_timeline():
    engine = AttackJourneyEngine()
    bari = BARIEngine()

    src_ip = "192.168.1.50"

    # Step 1: HTTP
    res1 = engine.process_flow_journey(
        src_ip=src_ip, dst_port=80, protocol=6, risk_score=0.15,
        bari_res={"bari_score": 0.20, "service_sensitivity": 0.10},
        level_str="Level 1 — Normal", bari_engine=bari
    )
    assert res1["journey_path"] == ["HTTP(80)"]
    assert len(res1["timeline"]) == 1

    # Step 2: SSH (Level 2 Early Warning)
    res2 = engine.process_flow_journey(
        src_ip=src_ip, dst_port=22, protocol=6, risk_score=0.48,
        bari_res={"bari_score": 0.52, "service_sensitivity": 0.65, "surface_progression": 0.65},
        level_str="Level 2 — Suspicious / Emerging", bari_engine=bari
    )
    assert res2["journey_path"] == ["HTTP(80)", "SSH(22)"]
    assert len(res2["timeline"]) == 2
    assert res2["detection_lead_time_sec"] >= 0.0

    # Step 3: MySQL (Level 3 Critical)
    res3 = engine.process_flow_journey(
        src_ip=src_ip, dst_port=3306, protocol=6, risk_score=0.92,
        bari_res={"bari_score": 0.88, "service_sensitivity": 0.75, "surface_progression": 0.90, "route_persistence": 0.80},
        level_str="Level 3 — High Risk", bari_engine=bari
    )

    assert res3["journey_path"] == ["HTTP(80)", "SSH(22)", "MySQL(3306)"]
    assert res3["attack_objective"]["category"] == "DATA_EXFILTRATION"
    assert res3["threat_priority_tier"] == "HIGH"


def test_attack_objective_inference():
    engine = AttackJourneyEngine()
    db_obj = engine.infer_attack_objective(["HTTP(80)"], 3306, 0.75)
    assert db_obj["category"] == "DATA_EXFILTRATION"
    assert db_obj["confidence"] > 85.0

    c2_obj = engine.infer_attack_objective(["HTTP(80)"], 4444, 1.00)
    assert c2_obj["category"] == "COMMAND_AND_CONTROL"
    assert c2_obj["confidence"] >= 95.0


def test_threat_priority_score():
    engine = AttackJourneyEngine()
    tps_high = engine.compute_threat_priority_score(
        risk_score=0.95, sensitivity=0.85, progression=0.90, persistence=0.80
    )
    assert tps_high > 80.0

    tps_low = engine.compute_threat_priority_score(
        risk_score=0.10, sensitivity=0.10, progression=0.10, persistence=0.10
    )
    assert tps_low < 30.0


def test_explainable_risk_breakdown():
    engine = AttackJourneyEngine()
    bari_res = {
        "route_novelty": 0.80,
        "route_deviation": 0.75,
        "surface_progression": 0.90,
        "route_acceleration": 0.60,
        "route_persistence": 0.70,
    }
    breakdown = engine.compute_explainable_risk_breakdown(bari_res, gnn_risk=0.85)
    assert "gnn_risk_contribution" in breakdown
    assert len(breakdown["breakdown_list"]) == 6
    assert breakdown["total_risk_score"] > 0.50


def test_attack_story_and_fingerprint():
    engine = AttackJourneyEngine()
    story = engine.generate_attack_story(
        src_ip="10.0.0.21", dst_port=3306, level_str="Level 3 — High Risk",
        bari_res={"route_novelty": 0.85}, objective_info={"objective": "Database Exfiltration", "confidence": 92.0},
        lead_time_sec=15.0, route=["HTTP(80)", "MySQL(3306)"], first_t_str="15:20:00"
    )
    assert "Host 10.0.0.21" in story
    assert "Database Exfiltration" in story

    fp = engine.compute_attack_fingerprint(
        {"surface_progression": 0.85, "route_deviation": 0.80}, ["HTTP(80)", "MySQL(3306)"]
    )
    assert fp["archetype_id"] == "ARCH-01-LATERAL"


def test_attack_path_simulator():
    engine = AttackJourneyEngine()
    bari = BARIEngine()

    sim = engine.simulate_attack_path(
        start_host="10.0.0.21", target_port=3306, strategy="lateral_movement", bari_engine=bari
    )
    assert sim["predicted_final_risk"] >= 0.75
    assert sim["early_warning_step"] is not None
    assert sim["predicted_lead_time_sec"] > 0.0
    assert len(sim["forecasted_next_targets"]) > 0
