"""
routes/auth_routes.py
=====================
Modular Flask Blueprint for Authentication and CSRF Protection.
"""

from flask import Blueprint, request, jsonify, make_response
from modules.auth_manager import (
    create_admin_session,
    validate_session,
    terminate_session,
    generate_csrf_token_for_session,
    verify_admin_request,
    verify_read_viewer_access,
    get_admin_token,
)
import hmac

auth_bp = Blueprint("auth_bp", __name__)


@auth_bp.route("/api/auth/csrf", methods=["GET"])
def api_auth_csrf():
    """Issue an independent CSRF token tied to active session or new visitor."""
    session_id = request.cookies.get("cybergraph_session")
    sess = validate_session(session_id)
    if sess:
        csrf_tok = sess.get("csrf_token")
    else:
        csrf_tok = generate_csrf_token_for_session("visitor")

    resp = make_response(jsonify({"status": "SUCCESS", "csrf_token": csrf_tok}))
    return resp


@auth_bp.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    """Authenticate administrator and issue HttpOnly, SameSite=Strict session cookie."""
    data = request.get_json(silent=True) or {}
    token = data.get("token") or data.get("password") or data.get("admin_token")
    if not token and request.form:
        token = request.form.get("token") or request.form.get("password") or request.form.get("admin_token")

    if not token or not isinstance(token, str):
        return jsonify({"status": "ERROR", "message": "Missing credentials token"}), 400

    admin_token = get_admin_token()
    if not hmac.compare_digest(token.strip(), admin_token.strip()):
        return jsonify({"status": "ERROR", "message": "Invalid administrator token"}), 401

    session_id, csrf_token = create_admin_session(ip=request.remote_addr or "127.0.0.1")

    resp = make_response(jsonify({
        "status": "SUCCESS",
        "message": "Administrator authenticated successfully",
        "csrf_token": csrf_token,
    }))

    is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"
    resp.set_cookie(
        "cybergraph_session",
        session_id,
        max_age=86400,
        httponly=True,
        samesite="Strict",
        secure=is_secure,
    )
    return resp


@auth_bp.route("/api/auth/logout", methods=["POST"])
def api_auth_logout():
    """Terminate administrator session and clear cookies."""
    session_id = request.cookies.get("cybergraph_session")
    terminate_session(session_id)
    resp = make_response(jsonify({"status": "SUCCESS", "message": "Logged out successfully"}))
    resp.delete_cookie("cybergraph_session")
    return resp


@auth_bp.route("/api/auth/status", methods=["GET"])
def api_auth_status():
    """Return current client authentication status and viewer privileges."""
    is_admin = verify_admin_request()
    is_viewer = verify_read_viewer_access()
    return jsonify({
        "status": "SUCCESS",
        "is_authenticated_admin": is_admin,
        "has_viewer_access": is_viewer or is_admin,
        "remote_addr": request.remote_addr,
    })

