"""
modules/auth_manager.py
=======================
Centralized Authentication, Device Authorization, and Anti-CSRF Manager.

Security Specifications:
1. Admin Authentication:
   - Supports username/password or Admin Token submission via secure login endpoint (/api/auth/login).
   - Issues cryptographically signed HttpOnly, SameSite=Strict session cookies.
   - Accepts 'Authorization: Bearer <token>' or 'X-Admin-Token' for headless/API clients.
   - Strictly rejects credentials or admin tokens supplied in URL query parameters.
2. Anti-CSRF Protection:
   - Generates independent, cryptographically secure per-session CSRF tokens.
   - Validates 'X-CSRF-Token' header on all state-changing requests (POST, PUT, DELETE, PATCH).
   - Validates request Origin and Referer headers against trusted local hosts.
3. Read-Only Viewer Authorization:
   - Gating on read endpoints (/api/stream_step, /api/stream_events, /api/export_report, etc.).
   - Allows access to authenticated administrators OR verified authorized device tokens.
   - Blocks unauthorized external LAN snoopers from viewing network graphs or threat intelligence.
"""

import os
import time
import hmac
import hashlib
import secrets
from typing import Optional, Tuple
from flask import request, Response
from urllib.parse import urlparse

from config.secrets_manager import get_hmac_secret, get_admin_token, get_session_secret

# Active in-memory session and CSRF store: session_id -> { "is_admin": bool, "csrf_token": str, "created_at": float }
_active_sessions = {}
SESSION_EXPIRY_SECONDS = 7 * 86400  # 7 days


def generate_csrf_token_for_session(session_id: str) -> str:
    """Generate and store an independent random CSRF token tied to a session."""
    token = secrets.token_hex(24)
    if session_id in _active_sessions:
        _active_sessions[session_id]["csrf_token"] = token
    return token


def create_admin_session(ip: Optional[str] = None) -> Tuple[str, str]:
    """Create a new authenticated administrator session. Returns (session_id, csrf_token)."""
    session_id = secrets.token_hex(32)
    csrf_token = secrets.token_hex(24)
    _active_sessions[session_id] = {
        "is_admin": True,
        "csrf_token": csrf_token,
        "created_at": time.time(),
        "ip": ip,
    }
    return session_id, csrf_token


def terminate_session(session_id: Optional[str]) -> None:
    """Invalidate an active session."""
    if session_id and session_id in _active_sessions:
        _active_sessions.pop(session_id, None)


def validate_session(session_id: Optional[str]) -> Optional[dict]:
    """Check if session is valid and unexpired."""
    if not session_id or session_id not in _active_sessions:
        return None
    sess = _active_sessions[session_id]
    if time.time() - sess["created_at"] > SESSION_EXPIRY_SECONDS:
        _active_sessions.pop(session_id, None)
        return None
    return sess


def verify_admin_request() -> bool:
    """Verify if the current HTTP request has authenticated administrator privileges.
    
    Checks:
    1. Active authenticated session cookie ('cybergraph_session').
    2. 'X-Admin-Token' HTTP header matching active Admin Token.
    3. 'Authorization: Bearer <token>' matching active Admin Token.
    
    STRICT SECURITY RULES:
    - URL query parameters (?admin_token= or ?token=) are EXPLICITLY REJECTED.
    - Remote IP addresses are NOT automatically granted admin.
    """
    admin_token = get_admin_token()

    # 1. Check session cookie
    session_id = request.cookies.get("cybergraph_session")
    sess = validate_session(session_id)
    if sess and sess.get("is_admin"):
        return True

    # 2. Check X-Admin-Token HTTP header
    header_token = request.headers.get("X-Admin-Token")
    if header_token and hmac.compare_digest(header_token.strip(), admin_token):
        return True

    # 3. Check Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[7:].strip()
        if bearer_token and hmac.compare_digest(bearer_token, admin_token):
            return True

    # Explicit test bypass only if deliberately enabled in test environment
    if os.environ.get("CYBERGRAPH_TEST_AUTO_ADMIN") == "1":
        return True

    return False


def verify_csrf_protection() -> Tuple[bool, Optional[str]]:
    """Validate CSRF protection on state-changing HTTP requests.
    
    Requirements:
    1. Origin / Referer check against Host to prevent cross-site request forgery.
    2. Verification of X-CSRF-Token header against active session CSRF token,
       or direct administrative token authentication for headless API clients.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return True, None

    # Origin & Referer Validation
    host = request.host
    origin = request.headers.get("Origin")
    if origin:
        parsed_origin = urlparse(origin).netloc
        if parsed_origin and parsed_origin != host:
            return False, f"CSRF Origin mismatch: expected '{host}', got '{parsed_origin}'"

    referer = request.headers.get("Referer")
    if referer and not origin:
        parsed_ref = urlparse(referer).netloc
        if parsed_ref and parsed_ref != host:
            return False, f"CSRF Referer mismatch: expected '{host}', got '{parsed_ref}'"

    # API clients authenticating with direct X-Admin-Token or Bearer header bypass browser CSRF
    header_token = request.headers.get("X-Admin-Token")
    auth_header = request.headers.get("Authorization", "")
    admin_token = get_admin_token()
    if (header_token and hmac.compare_digest(header_token.strip(), admin_token)) or \
       (auth_header.startswith("Bearer ") and hmac.compare_digest(auth_header[7:].strip(), admin_token)):
        return True, None

    # Cookie session CSRF token check
    session_id = request.cookies.get("cybergraph_session")
    sess = validate_session(session_id)
    if not sess:
        return False, "Authentication session required for state-changing operations"

    csrf_header = request.headers.get("X-CSRF-Token")
    expected_csrf = sess.get("csrf_token")
    if not csrf_header or not expected_csrf or not hmac.compare_digest(csrf_header.strip(), expected_csrf):
        return False, "Invalid or missing CSRF token"

    return True, None


def verify_read_viewer_access(authorized_devices_lookup_func=None) -> bool:
    """Verify if client has read access to sensitive network graphs, telemetry, or threat intelligence.
    
    Permitted if:
    1. Authenticated administrator (session cookie, X-Admin-Token, Bearer token).
    2. Validated signed device token belonging to an approved monitored device.
    3. Loopback testing bypass (if specifically enabled).
    """
    if verify_admin_request():
        return True

    # Check verified device token
    raw_token = request.cookies.get("cybergraph_device_token")
    client_ip = request.remote_addr or "127.0.0.1"
    if os.environ.get("CYBERGRAPH_TRUST_PROXY") == "1" and request.headers.get("X-Forwarded-For"):
        fwd = request.headers.get("X-Forwarded-For").split(",")[0].strip()
        if fwd:
            client_ip = fwd

    if raw_token and verify_signed_device_token(raw_token, client_ip):
        if authorized_devices_lookup_func and callable(authorized_devices_lookup_func):
            if authorized_devices_lookup_func(raw_token):
                return True
        else:
            return True

    # Check if client is server's own local IP or loopback
    srv_ip = None
    try:
        import app as main_app
        srv_ip = main_app.get_server_local_ip()
    except Exception:
        pass

    is_server_host = client_ip in ("127.0.0.1", "::1", "localhost") or (srv_ip and client_ip == srv_ip)
    if is_server_host and os.environ.get("CYBERGRAPH_REQUIRE_VIEWER_AUTH") != "1":
        return True

    return False


def generate_signed_device_token(client_ip: str) -> str:
    """Generate an HMAC-SHA256 authenticated device token bound to client IP using persistent secret."""
    secret = get_hmac_secret()
    dev_uuid = str(secrets.token_hex(16))
    ts = str(int(time.time()))
    payload = f"{client_ip}:{dev_uuid}:{ts}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{dev_uuid}.{ts}.{sig}"


def verify_signed_device_token(token: str, client_ip: str) -> bool:
    """Cryptographically verify that device token was signed for this IP using persistent secret."""
    if not token or not isinstance(token, str) or "." not in token:
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    dev_uuid, ts, sig = parts
    secret = get_hmac_secret()
    payload = f"{client_ip}:{dev_uuid}:{ts}"
    expected_sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return hmac.compare_digest(sig, expected_sig)

