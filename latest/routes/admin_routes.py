"""
routes/admin_routes.py
======================
Modular Flask Blueprint for OS Firewall and Administrative Blocking Actions.
"""

import logging
from flask import Blueprint, request, jsonify
from modules.auth_manager import (
    verify_csrf_protection,
    verify_admin_request,
    verify_read_viewer_access,
)

logger = logging.getLogger(__name__)

admin_bp = Blueprint("admin_bp", __name__)


def _actor():
    """Identify the requester in audit records (remote address + admin session marker)."""
    return f"admin@{request.remote_addr or 'unknown'}"


def _get_app_context():
    import sys
    if "__main__" in sys.modules and hasattr(sys.modules["__main__"], "fw_executor"):
        return sys.modules["__main__"]
    import app as main_app
    return main_app


@admin_bp.route("/api/execute_real_block", methods=["POST"])
def execute_real_block():
    main_app = _get_app_context()
    main_app.init_engines()

    csrf_ok, csrf_err = verify_csrf_protection()
    if not csrf_ok:
        return jsonify({"status": "ERROR", "message": f"Forbidden: {csrf_err}"}), 403

    if not verify_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized execute_real_block attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    target_ip = request.args.get("ip") or request.args.get("target_ip")
    if not target_ip and request.is_json:
        data = request.get_json(silent=True) or {}
        target_ip = data.get("ip") or data.get("target_ip")

    if not target_ip:
        return jsonify({"status": "ERROR", "message": "Missing IP parameter."}), 400

    clean_ip = main_app.sanitize_target_ip(target_ip)
    if not clean_ip:
        return jsonify({"status": "ERROR", "message": f"Invalid IP address: '{target_ip}'"}), 400

    body = request.get_json(silent=True) if request.is_json else None
    confirmed = bool((body or {}).get("confirm")) or request.args.get("confirm") in ("1", "true", "yes")
    fw = main_app.fw_executor
    if fw.is_enforcing and not confirmed:
        # Server-side guard: the UI shows a confirmation dialog and then sends confirm=true.
        return jsonify({
            "status": "CONFIRMATION_REQUIRED",
            "mode": fw.mode,
            "message": "Real OS blocking requires explicit confirmation (confirm=true).",
        }), 428

    res = fw.execute_block(clean_ip, allow_real=True, actor=_actor())
    if res.get("status") in ["REAL_FIREWALL_RULE_APPLIED", "REAL_FIREWALL_RULE_ELEVATED"] or res.get("execution_mode") in ["REAL_OS_FIREWALL", "REAL_OS_FIREWALL_ELEVATED"]:
        with main_app.state_lock:
            main_app.shared_state["stats"]["real_os_rules_applied"] += 1

    return jsonify(res)


@admin_bp.route("/api/unblock_ip", methods=["POST"])
def api_unblock_ip():
    main_app = _get_app_context()
    main_app.init_engines()

    csrf_ok, csrf_err = verify_csrf_protection()
    if not csrf_ok:
        return jsonify({"status": "ERROR", "message": f"Forbidden: {csrf_err}"}), 403

    if not verify_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized unblock_ip attempt from client IP: {request.remote_addr}")
        return jsonify({
            "status": "ERROR",
            "message": "Forbidden: Administrative access required"
        }), 403

    target_ip = request.args.get("ip") or request.args.get("target_ip")
    if not target_ip and request.is_json:
        data = request.get_json(silent=True) or {}
        target_ip = data.get("ip") or data.get("target_ip")

    if not target_ip:
        return jsonify({"status": "ERROR", "message": "Missing IP parameter."}), 400

    clean_ip = main_app.sanitize_target_ip(target_ip)
    if not clean_ip:
        return jsonify({"status": "ERROR", "message": f"Invalid IP address: '{target_ip}'"}), 400

    res = main_app.fw_executor.execute_unblock(clean_ip, actor=_actor())
    return jsonify(res)


@admin_bp.route("/api/verify_blocked", methods=["GET"])
def api_verify_blocked():
    main_app = _get_app_context()
    main_app.init_engines()

    target_ip = request.args.get("ip") or request.args.get("target_ip")
    if not target_ip:
        return jsonify({"status": "ERROR", "message": "Missing IP parameter."}), 400

    clean_ip = main_app.sanitize_target_ip(target_ip)
    if not clean_ip:
        return jsonify({"status": "ERROR", "message": f"Invalid IP address: '{target_ip}'"}), 400

    return jsonify(main_app.fw_executor.verify_blocked(clean_ip))


@admin_bp.route("/api/blocks/history", methods=["GET"])
def api_blocks_history():
    main_app = _get_app_context()
    main_app.init_engines()

    if not verify_read_viewer_access(lambda tok: tok in main_app._authorized_monitored_devices):
        return jsonify({"status": "ERROR", "message": "Viewer access required"}), 401

    fw_exec = main_app.fw_executor
    sim_blocked = list(fw_exec.simulated_acl_blocked_ips) if fw_exec else []
    real_blocked = list(fw_exec.active_os_firewall_rules.keys()) if fw_exec else []
    fw_history = list(fw_exec.action_history) if fw_exec and hasattr(fw_exec, "action_history") else []
    is_real_enabled = fw_exec.enable_real_blocking if fw_exec else False

    return jsonify({
        "status": "SUCCESS",
        "real_blocking_enabled": is_real_enabled,
        "simulated_blocked": sim_blocked,
        "real_blocked": real_blocked,
        "all_blocked": sorted(list(set(sim_blocked + real_blocked))),
        "history": fw_history[-50:]
    })



@admin_bp.route("/api/firewall/mode", methods=["GET"])
def api_firewall_mode_get():
    main_app = _get_app_context()
    main_app.init_engines()
    if not verify_read_viewer_access(lambda tok: tok in main_app._authorized_monitored_devices):
        return jsonify({"status": "ERROR", "message": "Viewer access required"}), 401
    fw = main_app.fw_executor
    return jsonify({
        "status": "SUCCESS",
        "mode": fw.mode,
        "enforcing": fw.is_enforcing,
        "active_real_rules": sorted(fw.active_os_firewall_rules),
    })


@admin_bp.route("/api/firewall/mode", methods=["POST"])
def api_firewall_mode_set():
    """Switch dry-run <-> enforce. Enforce additionally needs confirm == "ENFORCE"."""
    main_app = _get_app_context()
    main_app.init_engines()

    csrf_ok, csrf_err = verify_csrf_protection()
    if not csrf_ok:
        return jsonify({"status": "ERROR", "message": f"Forbidden: {csrf_err}"}), 403
    if not verify_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized firewall mode change from {request.remote_addr}")
        return jsonify({"status": "ERROR", "message": "Forbidden: Administrative access required"}), 403

    data = request.get_json(silent=True) or {}
    mode = str(data.get("mode", "")).strip().lower()
    if mode not in ("dry-run", "enforce"):
        return jsonify({"status": "ERROR", "message": "mode must be 'dry-run' or 'enforce'"}), 400
    if mode == "enforce" and data.get("confirm") != "ENFORCE":
        return jsonify({
            "status": "CONFIRMATION_REQUIRED",
            "message": "Enabling enforce mode requires confirm='ENFORCE'.",
        }), 428
    try:
        res = main_app.fw_executor.set_mode(mode, actor=_actor())
    except ValueError as exc:
        return jsonify({"status": "ERROR", "message": str(exc)}), 409
    return jsonify({"status": "SUCCESS", **res})
