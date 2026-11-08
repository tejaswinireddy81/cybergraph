"""
routes/stream_routes.py
=======================
Modular Flask Blueprint for Live Traffic Ingestion, Graph State Stream, and MITRE Mapping.
"""

import copy
import logging
from flask import Blueprint, request, jsonify
from modules.auth_manager import (
    verify_csrf_protection,
    verify_admin_request,
    verify_read_viewer_access,
)

logger = logging.getLogger(__name__)

stream_bp = Blueprint("stream_bp", __name__)


def _get_app_context():
    import app as main_app
    return main_app


@stream_bp.route("/api/stream_step", methods=["GET"])
def stream_step():
    """Returns a read-only snapshot of the shared server state (Viewer Access Protected)."""
    main_app = _get_app_context()
    main_app.init_engines()
    if not verify_read_viewer_access(lambda tok: tok in main_app._authorized_monitored_devices):
        return jsonify({"status": "ERROR", "message": "Unauthorized: Viewer access required"}), 401
    with main_app.state_lock:
        snapshot = copy.deepcopy(main_app.shared_state)
    return jsonify(snapshot)


@stream_bp.route("/api/stream_events", methods=["GET"])
def stream_events():
    """Returns recent evaluation threat events (Viewer Access Protected)."""
    main_app = _get_app_context()
    main_app.init_engines()
    if not verify_read_viewer_access(lambda tok: tok in main_app._authorized_monitored_devices):
        return jsonify({"status": "ERROR", "message": "Unauthorized: Viewer access required"}), 401
    with main_app.state_lock:
        events = list(main_app._recent_event_history)
    return jsonify({"events": events, "count": len(events)})


@stream_bp.route("/api/pipeline_diag", methods=["GET"])
def pipeline_diag():
    """Returns the last processed live flows with diagnostic breakdown (Viewer Access Protected)."""
    main_app = _get_app_context()
    main_app.init_engines()
    if not verify_read_viewer_access(lambda tok: tok in main_app._authorized_monitored_devices):
        return jsonify({"status": "ERROR", "message": "Unauthorized: Viewer access required"}), 401
    from modules.live_traffic_engine import _diag_flows
    return jsonify({
        "description": "Last live flows processed through the ML pipeline",
        "flows": list(_diag_flows),
        "flow_count": len(_diag_flows),
    })


@stream_bp.route("/api/pipeline_explain", methods=["GET"])
def pipeline_explain():
    """Returns the last flow's pipeline step breakdown and SHAP explanation (Viewer Access Protected)."""
    main_app = _get_app_context()
    main_app.init_engines()
    if not verify_read_viewer_access(lambda tok: tok in main_app._authorized_monitored_devices):
        return jsonify({"status": "ERROR", "message": "Unauthorized: Viewer access required"}), 401

    state = dict(main_app._last_pipeline_state)
    explain_state = dict(main_app._last_explain_state)

    xai_report = None
    if explain_state and "raw_feat_1d" in explain_state:
        try:
            xai_report = main_app.xai_engine.explain_flow_edge(
                edge_attr=explain_state["edge_attr"],
                risk_score=explain_state["risk_score"],
                source_host=explain_state["source_host"],
                target_service=explain_state["target_service"],
                raw_feat_1d=explain_state["raw_feat_1d"],
                xgb_model=explain_state["xgb_model"],
                feature_cols=None,
            )
        except Exception:
            pass

    mitre_explanations = []
    if state.get("mitre_mappings_count", 0) > 0:
        mitre_explanations.append(f"{state['mitre_mappings_count']} MITRE ATT&CK technique(s) mapped from behavioral evidence")

    return jsonify({
        "pipeline_state": state,
        "xai_explanation": xai_report,
        "mitre_explanations": mitre_explanations,
        "bari_context": {
            "route_string": "N/A (per-flow data)",
            "status": "BARI contextual enrichment active",
        }
    })


@stream_bp.route("/api/set_mode", methods=["POST"])
def set_mode():
    """Switch pipeline operating mode (CSRF & Admin Protected)."""
    main_app = _get_app_context()
    main_app.init_engines()

    csrf_ok, csrf_err = verify_csrf_protection()
    if not csrf_ok:
        return jsonify({"status": "ERROR", "message": f"Forbidden: {csrf_err}"}), 403

    if not verify_admin_request():
        logger.warning(f"[SECURITY 403] Unauthorized set_mode attempt from client IP: {request.remote_addr}")
        return jsonify({"status": "ERROR", "message": "Forbidden: Administrative access required"}), 403

    data = request.get_json(silent=True) or {}
    mode = data.get("mode") or request.args.get("mode")
    if not mode or mode not in ("normal", "attack", "offline"):
        mode = "normal"

    with main_app.state_lock:
        main_app.shared_state["mode"] = mode
        main_app.shared_state["state_version"] += 1
        v = main_app.shared_state["state_version"]

    logger.info(f"[GLOBAL MODE CHANGE] Server mode set to '{mode}' (state_version={v})")
    try:
        main_app._run_pipeline_cycle()
    except Exception as exc:
        logger.warning(f"Immediate cycle execution warning on mode change: {exc}")

    return jsonify({"status": "SUCCESS", "mode": mode, "state_version": v})


@stream_bp.route("/api/mitre_mappings", methods=["GET"])
def mitre_mappings():
    """Returns all supported MITRE ATT&CK technique mappings and metadata."""
    main_app = _get_app_context()
    main_app.init_engines()
    mapper = main_app.mitre_mapper
    supported = []
    for tech_id in mapper.get_supported_techniques():
        info = mapper.get_technique_info(tech_id)
        supported.append({
            "technique_id": info.technique_id,
            "technique_name": info.technique_name,
            "tactic": info.tactic,
            "description": info.description,
            "required_confidence_threshold": info.required_confidence_threshold,
            "evidence_weights": info.evidence_weights,
            "behavioral_indicators": info.behavioral_indicators,
        })
    return jsonify({
        "supported_techniques": supported,
        "unsupported_count": 17,
        "mapping_method": "evidence-based deterministic rules",
        "confidence_formula": "confidence = weighted evidence match × model confidence damping",
        "separation_of_concerns": "MITRE provides ATT&CK contextualization; XGBoost/AE/GNN/BARI detect and score behavior",
    })


@stream_bp.route("/api/threat_priorities", methods=["GET"])
def threat_priorities():
    """Returns ranked threat priority queue from Attack Journey Engine."""
    main_app = _get_app_context()
    main_app.init_engines()
    priorities = main_app.attack_journey_engine.get_top_threat_priorities(limit=10)
    for p in priorities:
        src_ip = p["source_ip"]
        correlation_report = main_app.correlation_engine._build_report(src_ip)
        mappings = main_app.mitre_mapper.map_flow_to_techniques(
            flow_record={"src_ip": src_ip},
            bari_result={},
            correlation_result=correlation_report,
            attack_journey={"journey_path": [], "attack_objective": {}}
        )
        p["mitre_techniques"] = [
            {
                "technique_id": m["technique_id"],
                "technique_name": m["technique_name"],
                "confidence": m["confidence"]
            }
            for m in mappings
        ]
    return jsonify({
        "priorities": priorities,
        "mitre_supported": ["T1046", "T1110", "T1498"],
        "total_active_threats": len(priorities),
    })


@stream_bp.route("/api/simulate_attack_path", methods=["GET"])
def simulate_attack_path():
    """Defensive 'What-If' Attack Path Simulator."""
    main_app = _get_app_context()
    main_app.init_engines()
    start_host = request.args.get("host", "10.0.0.21")
    target_port = int(request.args.get("port", 3306))
    strategy = request.args.get("strategy", "lateral_movement")
    sim = main_app.attack_journey_engine.simulate_attack_path(
        start_host=start_host, target_port=target_port,
        strategy=strategy, bari_engine=main_app.bari_engine
    )
    return jsonify(sim)
