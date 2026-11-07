"""
tests/test_bari_route_intelligence.py
=========================================
Unit and Integration Tests for Upgraded BARI Route Intelligence Module.
"""

import pytest
import numpy as np
import pandas as pd
from modules.bari_engine import BARIEngine


def test_bari_initialization():
    engine = BARIEngine()
    assert engine._baseline_learned is False
    assert len(engine._attacker_route_memory) == 0


def test_bari_baseline_learning():
    engine = BARIEngine()
    df_benign = pd.DataFrame({
        "L4_DST_PORT": [80, 80, 443, 80, 53, 22],
        "PROTOCOL": [6, 6, 6, 6, 17, 6],
        "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.2", "10.0.0.3", "10.0.0.1"]
    })
    engine.learn_normal_baseline(df_benign)
    assert engine._baseline_learned is True
    assert "80/6" in engine._benign_service_counts
    
    # Data-driven service sensitivity
    sens_80 = engine.get_service_sensitivity(80, 6)
    sens_22 = engine.get_service_sensitivity(22, 6)
    assert sens_80 < sens_22  # HTTP (more frequent) has lower sensitivity score than SSH


def test_bari_scoring_and_route_memory():
    engine = BARIEngine()
    df_benign = pd.DataFrame({
        "L4_DST_PORT": [80, 443, 80, 443],
        "PROTOCOL": [6, 6, 6, 6],
        "IPV4_SRC_ADDR": ["192.168.1.10", "192.168.1.10", "192.168.1.11", "192.168.1.11"]
    })
    engine.learn_normal_baseline(df_benign)

    src_ip = "192.168.1.100"
    
    # Flow 1: HTTP (normal web service)
    res1 = engine.score(src_ip=src_ip, dst_port=80, protocol=6, gnn_risk=0.10, timestamp=100.0)
    assert "bari_score" in res1
    assert "route_novelty" in res1
    assert "route_deviation" in res1
    assert "surface_progression" in res1
    assert "route_acceleration" in res1
    assert "route_persistence" in res1
    assert res1["bari_score"] < 0.40

    # Flow 2: Rapid access to SSH (Port 22) -> Route Acceleration & Surface Progression trigger
    res2 = engine.score(src_ip=src_ip, dst_port=22, protocol=6, gnn_risk=0.75, timestamp=100.3)
    assert res2["route_acceleration"] > 0.70
    assert res2["surface_progression"] > res1["surface_progression"]
    assert res2["bari_score"] > res1["bari_score"]

    # Flow 3: Access to Backdoor (Port 4444) -> Escalation
    res3 = engine.score(src_ip=src_ip, dst_port=4444, protocol=6, gnn_risk=0.95, timestamp=100.6)
    assert res3["bari_score"] > 0.70
    assert len(res3["observed_route"]) == 3
    assert "Backdoor/C2(4444)" in res3["route_string"]


def test_bari_ablation_flags():
    engine = BARIEngine()
    src_ip = "10.0.0.50"
    
    res_full = engine.score(src_ip=src_ip, dst_port=3389, protocol=6, gnn_risk=0.60, use_memory=True, use_progression=True)
    res_nomem = engine.score(src_ip=src_ip, dst_port=3389, protocol=6, gnn_risk=0.60, use_memory=False, use_progression=True)
    res_noprog = engine.score(src_ip=src_ip, dst_port=3389, protocol=6, gnn_risk=0.60, use_memory=True, use_progression=False)

    assert "bari_score" in res_full
    assert "bari_score" in res_nomem
    assert "bari_score" in res_noprog
