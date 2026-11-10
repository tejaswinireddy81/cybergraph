"""
routes/device_routes.py
=======================
Modular Flask Blueprint for Monitored Device Authorization and Access Control.
"""

import logging
from flask import Blueprint, request, jsonify
from modules.auth_manager import (
    verify_csrf_protection,
    verify_admin_request,
)

logger = logging.getLogger(__name__)

device_bp = Blueprint("device_bp", __name__)


def _get_app_context():
    import sys
    if "__main__" in sys.modules and hasattr(sys.modules["__main__"], "get_authorized_devices"):
        return sys.modules["__main__"]
    import app as main_app
    return main_app


@device_bp.route("/api/device_requests", methods=["GET"])
def api_device_requests():
    main_app = _get_app_context()
    main_app.init_engines()
    with main_app.state_lock:
        state_ver = main_app.shared_state.get("state_version", 0)
    auth_list = main_app.get_authorized_devices()
    pending_list = main_app.get_pending_requests()
    return jsonify({
        "status": "SUCCESS",
        "is_admin": main_app.is_admin_request(),
        "state_version": state_ver,
        "authorized_devices": auth_list,
        "pending_requests": pending_list,
    })


@device_bp.route("/api/approve_device", methods=["POST"])
def approve_device():
    main_app = _get_app_context()
    main_app.init_engines()

    csrf_ok, csrf_err = verify_csrf_protection()
    if not csrf_ok:
        return jsonify({"status": "ERROR", "message": f"Forbidden: {csrf_err}"}), 403

    if not verify_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized approve_device attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    req_id = request.args.get("request_id") or request.args.get("device_id") or request.args.get("ip")
    if not req_id and request.is_json:
        data = request.get_json(silent=True) or {}
        req_id = data.get("request_id") or data.get("device_id") or data.get("ip")

    if not req_id:
        return jsonify({"status": "ERROR", "message": "Missing request_id, device_id, or ip parameter"}), 400

    res = main_app.approve_device_request(req_id)
    if res:
        with main_app.state_lock:
            main_app.shared_state["state_version"] += 1
        return jsonify({"status": "SUCCESS", "message": f"Device request '{req_id}' approved", "approved_device": res})
    return jsonify({"status": "ERROR", "message": f"Pending request or device '{req_id}' not found"}), 404


@device_bp.route("/api/reject_device", methods=["POST"])
def reject_device():
    main_app = _get_app_context()
    main_app.init_engines()

    csrf_ok, csrf_err = verify_csrf_protection()
    if not csrf_ok:
        return jsonify({"status": "ERROR", "message": f"Forbidden: {csrf_err}"}), 403

    if not verify_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized reject_device attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    req_id = request.args.get("request_id") or request.args.get("device_id") or request.args.get("ip")
    if not req_id and request.is_json:
        data = request.get_json(silent=True) or {}
        req_id = data.get("request_id") or data.get("device_id") or data.get("ip")

    if not req_id:
        return jsonify({"status": "ERROR", "message": "Missing request_id, device_id, or ip parameter"}), 400

    ok = main_app.reject_device_request(req_id)
    if ok:
        with main_app.state_lock:
            main_app.shared_state["state_version"] += 1
        return jsonify({"status": "SUCCESS", "message": f"Device request '{req_id}' rejected"})
    return jsonify({"status": "ERROR", "message": f"Request '{req_id}' not found"}), 404


@device_bp.route("/api/revoke_device", methods=["POST"])
def revoke_device():
    main_app = _get_app_context()
    main_app.init_engines()

    csrf_ok, csrf_err = verify_csrf_protection()
    if not csrf_ok:
        return jsonify({"status": "ERROR", "message": f"Forbidden: {csrf_err}"}), 403

    if not verify_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized revoke_device attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    req_id = request.args.get("device_id") or request.args.get("ip") or request.args.get("request_id") or request.args.get("device_token")
    if not req_id and request.is_json:
        data = request.get_json(silent=True) or {}
        req_id = data.get("device_id") or data.get("ip") or data.get("request_id") or data.get("device_token")

    if not req_id:
        return jsonify({"status": "ERROR", "message": "Missing device_id, ip, or device_token parameter"}), 400

    ok = main_app.revoke_device_authorization(req_id)
    if ok:
        with main_app.state_lock:
            main_app.shared_state["state_version"] += 1
        return jsonify({"status": "SUCCESS", "message": f"Device '{req_id}' authorization revoked"})
    return jsonify({"status": "ERROR", "message": f"Device '{req_id}' not found or cannot be revoked"}), 404

