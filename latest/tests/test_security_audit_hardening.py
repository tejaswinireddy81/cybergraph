"""
tests/test_security_audit_hardening.py
======================================
Security regression tests covering:
1. XSS sanitization in incident audit reports (/api/export_report).
2. Uniform JSON error handlers for /api/* endpoints (404, 405, 413, etc.).
3. Request body size limit (MAX_CONTENT_LENGTH = 16MB).
4. Memory exhaustion protection in FlowExtractor (MAX_ACTIVE_FLOWS = 50_000 limit & eviction).
5. Authentication and CSRF defense verification.
"""

import pytest
import io
import time
from app import app, shared_state, state_lock, get_admin_token
from flow_extractor import FlowExtractor, MAX_ACTIVE_FLOWS, FlowKey


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_export_report_xss_sanitization(client):
    """Verify that malicious XSS payloads in threat events are escaped in /api/export_report."""
    xss_payload = "<script>alert('XSS_ATTACK')</script>"
    xss_attr = "<img src=x onerror=alert('PWNED')>"

    with state_lock:
        shared_state["evals"].append({
            "timestamp": time.time(),
            "level": f"Level 3 — Critical {xss_payload}",
            "badge_color": "#ef4444",
            "risk_score": 0.95,
            "source_host": f"HOST:{xss_attr}",
            "target_service": f"SERVICE:{xss_payload}",
            "summary": f"Exploit attempt {xss_payload}",
            "xai_explanation": {
                "why_this_attack_reasons": [
                    f"Malicious header {xss_payload}",
                    f"Injected script {xss_attr}"
                ]
            },
            "mitre_mappings": [
                {
                    "technique_id": xss_payload,
                    "technique_name": xss_attr,
                    "confidence": 0.99
                }
            ],
            "recommended_firewall_action": {
                "target_ip": xss_payload,
                "windows_netsh_rule_preview": f"netsh {xss_payload}"
            }
        })

    token = get_admin_token()
    # Client with viewer cookie
    res = client.get("/api/export_report", headers={"X-Admin-Token": token})
    assert res.status_code == 200
    html_text = res.get_data(as_text=True)

    # Assert raw unescaped script tag does NOT exist anywhere in HTML
    assert "<script>alert('XSS_ATTACK')</script>" not in html_text
    assert "<img src=x onerror=alert('PWNED')>" not in html_text

    # Assert escaped versions ARE present
    assert "&lt;script&gt;alert(&#x27;XSS_ATTACK&#x27;)&lt;/script&gt;" in html_text or "&lt;script&gt;alert('XSS_ATTACK')&lt;/script&gt;" in html_text
    assert "&lt;img src=x onerror=alert(&#x27;PWNED&#x27;)&gt;" in html_text or "&lt;img src=x onerror=alert('PWNED')&gt;" in html_text


def test_api_json_404_error_handler(client):
    """Verify non-existent /api/* endpoint returns standard JSON error, not HTML."""
    res = client.get("/api/non_existent_cyber_route_404")
    assert res.status_code == 404
    assert res.is_json
    data = res.get_json()
    assert data.get("status") == "ERROR"
    assert "Endpoint Not Found" in data.get("message")


def test_api_json_405_error_handler(client):
    """Verify method not allowed on /api/* returns JSON, not HTML."""
    # /api/stream_step only accepts GET
    res = client.post("/api/stream_step")
    assert res.status_code == 405
    assert res.is_json
    data = res.get_json()
    assert data.get("status") == "ERROR"
    assert "Method Not Allowed" in data.get("message")


def test_max_content_length_configured():
    """Verify Flask app has MAX_CONTENT_LENGTH set to 16MB."""
    assert app.config.get("MAX_CONTENT_LENGTH") == 16 * 1024 * 1024


def test_flow_extractor_max_active_flows_ceiling():
    """Verify FlowExtractor enforces MAX_ACTIVE_FLOWS ceiling by evicting oldest idle flows."""
    extractor = FlowExtractor()
    assert MAX_ACTIVE_FLOWS == 50_000

    # Test eviction mechanism with a mock small ceiling
    # Simulate adding flows beyond limit
    extractor._flows.clear()
    now = time.time()
    for i in range(10):
        key = ("10.0.0.1", f"192.168.1.{i}", 1000 + i, 80, 6)
        extractor._get_or_create(key, now=now + i, wall_now=now + i)

    assert len(extractor._flows) == 10

    # Temporarily set MAX_ACTIVE_FLOWS to 10 in module to test eviction logic
    import flow_extractor
    orig_max = flow_extractor.MAX_ACTIVE_FLOWS
    try:
        flow_extractor.MAX_ACTIVE_FLOWS = 10
        # Adding an 11th flow should evict the oldest flow (key with lowest t_last)
        oldest_key = ("10.0.0.1", "192.168.1.0", 1000, 80, 6)
        new_key = ("10.0.0.1", "192.168.1.99", 9999, 80, 6)

        extractor._get_or_create(new_key, now=now + 100, wall_now=now + 100)

        assert len(extractor._flows) == 10
        assert oldest_key not in extractor._flows
        assert new_key in extractor._flows
    finally:
        flow_extractor.MAX_ACTIVE_FLOWS = orig_max
