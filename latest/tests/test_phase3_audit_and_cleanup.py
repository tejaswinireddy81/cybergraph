"""
tests/test_phase3_audit_and_cleanup.py
======================================
Verification test suite for Phase 3:
- CG-03: Audit report contents & accuracy
- CG-01: Removal of raw AI model pipeline-stage values from frontend
- Preservation of internal model execution, risk scores, and diagnostics
"""

import time
import pytest
import app as app_module
from app import (
    app,
    init_engines,
    _run_pipeline_cycle,
    _recent_event_history,
    state_lock,
    set_network_profile_override,
)


@pytest.fixture(autouse=True)
def setup_phase3_env():
    init_engines()
    with state_lock:
        _recent_event_history.clear()
    set_network_profile_override({
        "iface": "Wi-Fi",
        "local_ip": "10.160.36.211",
        "local_subnet": "10.160.36.0/24",
        "default_gateway": "10.160.36.14",
    })
    yield


def test_frontend_does_not_display_unwanted_pipeline_stage_values():
    """Requirement: Unwanted AI model pipeline-stage table/values removed from frontend."""
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Assert raw table and stage headers are removed from dashboard
    assert "AI Model Pipeline Stage" not in html
    assert 'id="m-xgb"' not in html
    assert 'id="m-ae"' not in html
    assert 'id="m-fus"' not in html
    assert 'id="m-gnn"' not in html
    assert 'id="m-fin"' not in html


def test_internal_model_execution_and_endpoints_preserved():
    """Requirement: All ML models still execute internally and diagnostic endpoints remain operational."""
    client = app.test_client()

    # /api/stream_step returns pipeline_metrics for telemetry and internal state
    res_state = client.get("/api/stream_step")
    assert res_state.status_code == 200
    state_data = res_state.get_json()
    assert "pipeline_metrics" in state_data
    assert "xgb_prob" in state_data["pipeline_metrics"]
    assert "ae_score" in state_data["pipeline_metrics"]
    assert "fusion_score" in state_data["pipeline_metrics"]
    assert "gnn_risk" in state_data["pipeline_metrics"]

    # /api/pipeline_diag works
    res_diag = client.get("/api/pipeline_diag")
    assert res_diag.status_code == 200

    # /api/pipeline_explain works
    res_explain = client.get("/api/pipeline_explain")
    assert res_explain.status_code == 200


def test_audit_report_contains_all_required_sections_and_evidence():
    """Requirement: Audit report contains threat summary, timestamp, source/dest, evidence, actions, firewall status."""
    client = app.test_client()

    # Create synthetic threat event in history
    now_ts = time.time()
    synthetic_event = {
        "level": "Level 3 — High Risk",
        "risk_category": "CRITICAL_ATTACK_THREAT",
        "risk_score": 0.9452,
        "source_host": "HOST:10.160.36.186",
        "target_service": "SERVICE:445/6",
        "timestamp": now_ts,
        "summary": "CRITICAL ATTACK: [HOST:10.160.36.186 -> SERVICE:445/6] Risk: 0.9452",
        "badge_color": "#ef4444",
        "xai_explanation": {
            "why_this_attack_reasons": ["Unusually high byte volume (SHAP +0.42)", "Suspicious SMB port targeting"]
        },
        "mitre_mappings": [
            {"technique_id": "T1110", "technique_name": "Brute Force", "confidence": 0.88, "tactic": "Credential Access"}
        ],
        "recommended_firewall_action": {
            "target_ip": "10.160.36.186",
            "windows_netsh_rule_preview": "netsh advfirewall firewall add rule name=\"Block-GNN-10.160.36.186\" dir=in action=block remoteip=10.160.36.186",
        }
    }

    with state_lock:
        _recent_event_history.append(synthetic_event)

    # Perform a block and unblock to populate firewall action history
    app_module.fw_executor.execute_block("10.160.36.186")
    app_module.fw_executor.execute_unblock("10.160.36.186")
    app_module.fw_executor.execute_block("10.160.36.199")

    # Fetch JSON format of report
    res_json = client.get("/api/export_report?format=json")
    assert res_json.status_code == 200
    report_data = res_json.get_json()

    assert "telemetry_summary" in report_data
    assert "firewall_status" in report_data
    assert "10.160.36.199" in report_data["firewall_status"]["simulated_acl_blocked_ips"]
    assert len(report_data["firewall_status"]["action_history"]) >= 2
    assert report_data["detected_threats_count"] >= 1

    # Fetch HTML format of report
    res_html = client.get("/api/export_report")
    assert res_html.status_code == 200
    html_text = res_html.get_data(as_text=True)

    # Verify all required sections exist in the rendered HTML report
    assert "CYBERGRAPH COMMAND CENTER — Incident Security Audit Report" in html_text
    assert "Level 3 — High Risk" in html_text
    assert "HOST:10.160.36.186" in html_text
    assert "SERVICE:445/6" in html_text
    assert "Unusually high byte volume" in html_text
    assert "T1110 (Brute Force)" in html_text
    assert "Block IP 10.160.36.186 via OS Firewall Rule" in html_text
    assert "Active Firewall & Containment Status" in html_text
    assert "Firewall Action & Response History" in html_text

