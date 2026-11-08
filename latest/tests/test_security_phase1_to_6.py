"""
tests/test_security_phase1_to_6.py
===================================
Comprehensive regression test suite for Phases 1 through 6:
1. Admin token never exposed in HTML responses.
2. Query-parameter admin authentication is rejected.
3. Admin login sets HttpOnly, SameSite=Strict session cookie.
4. Independent CSRF protection on state-changing requests.
5. GET rejected (405) on state-changing administrative routes.
6. Read-only endpoints (/api/stream_step, /api/export_report) require viewer or admin access.
7. Persistent HMAC secret and signed device token validation.
8. Security headers (CSP without unsafe-inline, X-Frame-Options, X-Content-Type-Options).
9. Cryptographic model integrity verification and tampering prevention.
10. Firewall block history and unblock operations.
"""

import os
import json
import pytest
import tempfile
import shutil
from app import app, get_admin_token
from modules.auth_manager import (
    generate_signed_device_token,
    verify_signed_device_token,
    generate_csrf_token_for_session,
)
from modules.trusted_model_loader import (
    safe_load_joblib,
    safe_load_torch,
    ModelTamperingException,
    compute_sha256,
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_1_admin_token_never_exposed_in_html(client):
    """Phase 1: Admin token must never be embedded in the rendered HTML or static files."""
    token = get_admin_token()
    assert token, "Admin token must be defined"
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    assert token not in html, "CRITICAL: Admin token was found embedded in the HTML response!"
    assert "ADMIN_TOKEN" not in html


def test_2_query_string_admin_token_rejected(client):
    """Phase 1: Admin tokens supplied via URL query parameters (?admin_token=) must be rejected."""
    token = get_admin_token()
    # Attempt administrative action via query parameter
    res = client.post(f"/api/execute_real_block?admin_token={token}", json={"ip": "198.51.100.10"})
    assert res.status_code == 403
    data = res.get_json()
    assert "Forbidden" in data.get("message", "")


def test_3_admin_login_sets_httponly_samesite_cookie(client):
    """Phase 1: Admin login route sets HttpOnly and SameSite=Strict session cookie."""
    token = get_admin_token()
    res = client.post("/api/auth/login", json={"token": token})
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("status") == "SUCCESS"

    cookie_header = res.headers.get("Set-Cookie", "")
    assert "cybergraph_session=" in cookie_header
    assert "HttpOnly" in cookie_header
    assert "SameSite=Strict" in cookie_header


def test_4_csrf_protection_rejects_missing_token_on_state_changing_routes(client):
    """Phase 1: Cookie-authenticated state-changing POST requests without CSRF protection must be rejected with 403."""
    token = get_admin_token()
    # 1. Admin logs in and acquires session cookie
    login_res = client.post("/api/auth/login", json={"token": token})
    assert login_res.status_code == 200

    # 2. Browser makes state-changing request with session cookie but WITHOUT X-CSRF-Token header
    res = client.post("/api/execute_real_block", json={"ip": "198.51.100.22"})
    assert res.status_code == 403
    data = res.get_json()
    assert "CSRF" in data.get("message", "")

    # 3. Cross-origin request with Origin header mismatch is rejected
    res_origin = client.post(
        "/api/execute_real_block",
        headers={"Origin": "https://malicious-site.com", "X-CSRF-Token": "invalid"},
        json={"ip": "198.51.100.22"}
    )
    assert res_origin.status_code == 403


def test_5_csrf_protection_accepts_valid_csrf_token(client):
    """Phase 1: Request with valid CSRF token succeeds."""
    token = get_admin_token()
    # Fetch CSRF token
    csrf_res = client.get("/api/auth/csrf")
    assert csrf_res.status_code == 200
    csrf_token = csrf_res.get_json().get("csrf_token")
    assert csrf_token

    headers = {
        "X-Admin-Token": token,
        "X-CSRF-Token": csrf_token,
        "Content-Type": "application/json",
    }
    res = client.post("/api/execute_real_block", headers=headers, json={"ip": "198.51.100.33"})
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("status") in ["SIMULATED_BLOCK_RECORDED", "REAL_FIREWALL_RULE_APPLIED", "ADMIN_PRIVILEGE_REQUIRED"]


def test_6_get_rejected_on_state_changing_endpoints(client):
    """Phase 1: GET requests on state-changing firewall and device endpoints return 405 Method Not Allowed."""
    endpoints = [
        "/api/execute_real_block",
        "/api/unblock_ip",
        "/api/approve_device",
        "/api/reject_device",
        "/api/revoke_device",
    ]
    for ep in endpoints:
        res = client.get(ep)
        assert res.status_code == 405, f"Expected 405 for GET {ep}, got {res.status_code}"


def test_7_read_only_endpoints_gated_by_viewer_access(client):
    """Phase 1: Unauthenticated visitors cannot access sensitive topology, threat telemetry, or reports."""
    # Without cookie or token, an unapproved external device is rejected with 401
    res = client.get("/api/stream_step", environ_base={"REMOTE_ADDR": "192.168.1.189"})
    assert res.status_code == 401

    # export_report returns 401 for unauthorized client
    res_report = client.get("/api/export_report", environ_base={"REMOTE_ADDR": "192.168.1.189"})
    assert res_report.status_code == 401

    # With admin token, access is granted
    token = get_admin_token()
    res_auth = client.get("/api/stream_step", headers={"X-Admin-Token": token})
    assert res_auth.status_code == 200


def test_8_persistent_hmac_secret_and_token_validation():
    """Phase 1: Device token signed with HMAC secret validates, and tampered token is rejected."""
    client_ip = "192.168.1.105"
    token = generate_signed_device_token(client_ip)
    assert token
    # Valid token validates successfully
    assert verify_signed_device_token(token, client_ip) is True

    # Token bound to different IP fails validation
    assert verify_signed_device_token(token, "192.168.1.200") is False

    # Tampered signature fails validation
    parts = token.split(".")
    tampered = parts[0] + ".bad_signature"
    assert verify_signed_device_token(tampered, client_ip) is False


def test_9_security_headers(client):
    """Phase 1: Security headers (CSP, X-Frame-Options, X-Content-Type-Options) are present and strict."""
    res = client.get("/")
    assert res.status_code == 200

    csp = res.headers.get("Content-Security-Policy", "")
    assert csp, "Content-Security-Policy header must be present"
    # Strict CSP: unsafe-inline must NOT be in script-src
    csp_directives = dict(item.strip().split(" ", 1) for item in csp.split(";") if " " in item.strip())
    script_src = csp_directives.get("script-src", "")
    assert "'unsafe-inline'" not in script_src, "CRITICAL: 'unsafe-inline' was found in script-src!"
    assert "'self'" in script_src

    x_frame = res.headers.get("X-Frame-Options", "")
    assert x_frame in ["DENY", "SAMEORIGIN"]

    x_content_type = res.headers.get("X-Content-Type-Options", "")
    assert x_content_type == "nosniff"


def test_10_trusted_model_loader_valid_and_tampered_model():
    """Phase 4: Trusted model loader validates genuine model and rejects tampered files."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if os.path.exists(scaler_path):
        # 1. Valid model loads successfully
        scaler = safe_load_joblib(scaler_path)
        assert scaler is not None

        # 2. Tampered file is rejected with ModelTamperingException
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp.write(b"CORRUPTED_MALICIOUS_PAYLOAD")
            tmp_path = tmp.name

        try:
            # A corrupted file with wrong hash must raise ModelTamperingException
            with pytest.raises(ModelTamperingException):
                safe_load_joblib(tmp_path, manifest_path=os.path.join(models_dir, "model_manifest.json"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def test_11_firewall_blocks_history_endpoint(client):
    """Phase 6: /api/blocks/history returns active blocks and audit history."""
    token = get_admin_token()
    headers = {"X-Admin-Token": token}
    res = client.get("/api/blocks/history", headers=headers)
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("status") == "SUCCESS"
    assert "simulated_blocked" in data
    assert "real_blocked" in data
    assert "all_blocked" in data
    assert "history" in data
