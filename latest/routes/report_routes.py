"""
routes/report_routes.py
=======================
Modular Flask Blueprint for Export Reports and Multi-Hop Attack Journeys.
"""

import copy
import time
from flask import Blueprint, request, jsonify, Response
from modules.auth_manager import verify_read_viewer_access

report_bp = Blueprint("report_bp", __name__)


def _get_app_context():
    import app as main_app
    return main_app


@report_bp.route("/api/attack_journey/<src_ip>", methods=["GET"])
def attack_journey(src_ip):
    """Returns the multi-hop attack journey for a specific source IP (Viewer Protected)."""
    main_app = _get_app_context()
    main_app.init_engines()
    if not verify_read_viewer_access(lambda tok: tok in main_app._authorized_monitored_devices):
        return jsonify({"status": "ERROR", "message": "Unauthorized: Viewer access required"}), 401

    timeline = main_app.correlation_engine.get_timeline(src_ip)
    journey = main_app.attack_journey_engine._journeys.get(src_ip)
    top_priority = main_app.attack_journey_engine.get_top_threat_priorities(limit=1)

    if journey and journey["timeline"]:
        return jsonify({
            "source_ip": src_ip,
            "journey_path": journey["journey_path"],
            "route_string": journey["route_string"],
            "timeline": journey["timeline"][-10:],
            "attack_objective": journey.get("timeline", [{}])[-1].get("stage", "Unknown"),
            "threat_priority_score": top_priority[0]["threat_priority_score"] if top_priority else 0,
            "threat_priority_tier": top_priority[0]["threat_priority_tier"] if top_priority else "LOW",
            "detection_lead_time_sec": journey.get("detection_lead_time_sec", 0),
            "early_detection_time": journey.get("early_t"),
            "critical_detection_time": journey.get("critical_t"),
        })
    return jsonify({
        "source_ip": src_ip,
        "message": "No attack journey data found for this IP.",
        "timeline": timeline,
    })


@report_bp.route("/api/export_report", methods=["GET"])
def export_report():
    main_app = _get_app_context()
    main_app.init_engines()
    if not verify_read_viewer_access(lambda tok: tok in main_app._authorized_monitored_devices):
        return Response("<h1>401 Unauthorized: Viewer access required</h1>", mimetype="text/html", status=401)

    fw_exec = main_app.fw_executor
    with main_app.state_lock:
        st = copy.deepcopy(main_app.shared_state["stats"])
        recent_events = copy.deepcopy(main_app._recent_event_history)
        active_evals = copy.deepcopy(main_app.shared_state.get("evals", []))
        sim_blocked = list(fw_exec.simulated_acl_blocked_ips) if fw_exec else []
        real_blocked = list(fw_exec.active_os_firewall_rules.keys()) if fw_exec else []
        fw_history = list(fw_exec.action_history) if fw_exec and hasattr(fw_exec, "action_history") else []
        is_real_enabled = fw_exec.enable_real_blocking if fw_exec else False

    threat_records = []
    seen_ids = set()
    all_raw_events = recent_events + active_evals
    for ev in all_raw_events:
        rec_id = (ev.get("source_host"), ev.get("target_service"), ev.get("timestamp"))
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            threat_records.append(ev)

    threat_records.sort(key=lambda x: (x.get("risk_score", 0.0), x.get("timestamp", 0.0)), reverse=True)

    if request.args.get("format") == "json":
        return jsonify({
            "status": "SUCCESS",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "telemetry_summary": st,
            "firewall_status": {
                "real_blocking_enabled": is_real_enabled,
                "simulated_acl_blocked_ips": sim_blocked,
                "real_os_firewall_blocked_ips": real_blocked,
                "total_blocked_count": len(set(sim_blocked + real_blocked)),
                "action_history": fw_history,
            },
            "detected_threats_count": len(threat_records),
            "threat_events": threat_records,
        })

    # Build HTML Incident Rows
    threat_rows_html = ""
    if threat_records:
        for idx, ev in enumerate(threat_records[:50], 1):
            ts_val = ev.get("timestamp")
            if ts_val:
                try:
                    ts_float = float(ts_val)
                    if ts_float > 1e11:
                        ts_float /= 1000.0
                    ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_float))
                except Exception:
                    ts_str = str(ts_val)
            else:
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S")

            lvl = ev.get("level", "Level 1 — Normal")
            badge_color = ev.get("badge_color", "#10b981")
            risk_score = float(ev.get("risk_score", 0.0))
            src = ev.get("source_host", "UNKNOWN")
            dst = ev.get("target_service", "UNKNOWN")
            summary = ev.get("summary", "Normal Traffic Flow")

            xai = ev.get("xai_explanation", {})
            reasons = xai.get("why_this_attack_reasons", []) if isinstance(xai, dict) else []
            reasons_html = "".join([f"<li>{r}</li>" for r in reasons]) if reasons else "<li>Baseline traffic profile match</li>"

            mitre_list = ev.get("mitre_mappings", [])
            mitre_html = ""
            if mitre_list:
                mitre_badges = []
                for m in mitre_list:
                    mitre_badges.append(f"<span style='background:#e0f2fe; color:#0369a1; padding:2px 6px; border-radius:4px; font-weight:700;'>{m.get('technique_id')} ({m.get('technique_name')}) [{(float(m.get('confidence',0))*100):.1f}%]</span>")
                mitre_html = f"<div style='margin-top:4px;'><strong>MITRE ATT&CK:</strong> {' '.join(mitre_badges)}</div>"

            rec_fw = ev.get("recommended_firewall_action", {})
            target_ip = rec_fw.get("target_ip", src.replace("HOST:", "").replace("EXT:", "").replace("GW:", "").replace("PENDING:", ""))
            cmd_preview = rec_fw.get("windows_netsh_rule_preview", f"netsh advfirewall firewall add rule name=\"Block-GNN-{target_ip}\" dir=in action=block remoteip={target_ip}")
            rec_action = f"Block IP {target_ip} via OS Firewall Rule<br><code style='background:#f1f5f9; padding:2px 4px; font-size:11px;'>{cmd_preview}</code>" if risk_score >= 0.50 else "Monitor traffic; no blocking action required."

            is_ip_blocked = (target_ip in sim_blocked) or (target_ip in real_blocked)
            block_status_html = f"<span style='color:#ef4444; font-weight:700;'>BLOCKED</span>" if is_ip_blocked else "<span style='color:#10b981; font-weight:700;'>ALLOWED / MONITORING</span>"

            threat_rows_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-family: monospace; font-size: 11px;">{ts_str}</td>
                <td style="padding: 10px;"><span style="background:{badge_color}; color:#fff; padding:3px 8px; border-radius:4px; font-weight:700; font-size:11px;">{lvl}</span><br><small>Risk: {risk_score:.4f}</small></td>
                <td style="padding: 10px;"><strong>Src:</strong> {src}<br><strong>Dst:</strong> {dst}</td>
                <td style="padding: 10px; font-size: 12px;">
                    <div>{summary}</div>
                    <ul style="margin:4px 0 0 16px; padding:0; font-size:11px; color:#475569;">{reasons_html}</ul>
                    {mitre_html}
                </td>
                <td style="padding: 10px; font-size: 11px;">{rec_action}</td>
                <td style="padding: 10px; font-size: 11px;">{block_status_html}</td>
            </tr>
            """
    else:
        threat_rows_html = "<tr><td colspan='6' style='text-align:center; padding:20px; color:#64748b;'>No security incidents recorded in current session.</td></tr>"

    fw_rows_html = ""
    if fw_history:
        for act in reversed(fw_history[-30:]):
            act_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(act.get("timestamp", time.time())))
            act_type = act.get("action", "UNKNOWN")
            act_ip = act.get("ip", "-")
            act_mode = act.get("execution_mode", "SIMULATED_MEMORY_ONLY")
            act_status = act.get("status", "-")
            fw_rows_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 8px; font-family: monospace; font-size: 11px;">{act_ts}</td>
                <td style="padding: 8px; font-weight:700; color:{'#ef4444' if act_type == 'BLOCK' else '#10b981'};">{act_type}</td>
                <td style="padding: 8px; font-family: monospace;">{act_ip}</td>
                <td style="padding: 8px; font-size: 11px;">{act_mode}</td>
                <td style="padding: 8px; font-size: 11px;">{act_status}</td>
            </tr>
            """
    else:
        fw_rows_html = "<tr><td colspan='5' style='text-align:center; padding:15px; color:#64748b;'>No firewall actions executed in current session.</td></tr>"

    blocked_list_str = ", ".join(sorted(list(set(sim_blocked + real_blocked)))) if (sim_blocked or real_blocked) else "None"

    report_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CYBERGRAPH COMMAND CENTER — Incident Security Audit Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #1e293b; line-height: 1.5; background: #f8fafc; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
        h1 {{ color: #0f172a; border-bottom: 3px solid #0284c7; padding-bottom: 12px; margin-top: 0; font-size: 26px; }}
        h2 {{ color: #1e293b; font-size: 18px; margin-top: 24px; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; }}
        .meta-bar {{ display: flex; justify-content: space-between; font-size: 13px; color: #64748b; margin-bottom: 20px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }}
        .kpi-box {{ background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; text-align: center; }}
        .kpi-val {{ font-size: 24px; font-weight: 800; color: #0f172a; margin-top: 4px; }}
        .kpi-lbl {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        th {{ background: #f8fafc; padding: 10px; text-align: left; font-weight: 700; color: #475569; border-bottom: 2px solid #cbd5e1; }}
        td {{ vertical-align: top; }}
        .status-box {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 14px; margin-bottom: 20px; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CYBERGRAPH COMMAND CENTER — Incident Security Audit Report</h1>
        <div class="meta-bar">
            <span><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</span>
            <span><strong>Active Source:</strong> {st['current_data_source']}</span>
            <span><strong>Auditor Mode:</strong> Real-Time Automated Defense Inspection</span>
        </div>
        <div class="kpi-grid">
            <div class="kpi-box"><div class="kpi-lbl">Total Evaluated Flows</div><div class="kpi-val">{st['total_flows']}</div></div>
            <div class="kpi-box"><div class="kpi-lbl">Level 1 (Normal)</div><div class="kpi-val" style="color: #10b981;">{st['level1_normal']}</div></div>
            <div class="kpi-box"><div class="kpi-lbl">Level 2 (Suspicious)</div><div class="kpi-val" style="color: #f59e0b;">{st['level2_suspicious']}</div></div>
            <div class="kpi-box"><div class="kpi-lbl">Level 3 (Critical)</div><div class="kpi-val" style="color: #ef4444;">{st['level3_critical']}</div></div>
            <div class="kpi-box"><div class="kpi-lbl">Peak Risk Observed</div><div class="kpi-val" style="color: #6366f1;">{st.get('peak_risk', 0.0):.4f}</div></div>
            <div class="kpi-box"><div class="kpi-lbl">Firewall Rules Applied</div><div class="kpi-val" style="color: #0284c7;">{st['real_os_rules_applied']}</div></div>
        </div>
        <div class="status-box">
            <h3 style="margin-top:0; color:#1e40af; font-size:15px;">Active Firewall & Containment Status</h3>
            <div><strong>Enforcement Mode:</strong> {'REAL OS FIREWALL' if is_real_enabled else 'SIMULATED ACL / RECOMMENDED APPROVAL'}</div>
            <div><strong>Currently Blocked IP Addresses:</strong> {blocked_list_str}</div>
            <div><strong>Real Windows Firewall Rules Active:</strong> {len(real_blocked)} | <strong>Simulated In-Memory Blocks:</strong> {len(sim_blocked)}</div>
        </div>
        <h2>1. Detected Threat Events & Diagnostic Findings</h2>
        <table>
            <thead><tr><th style="width:140px;">Timestamp</th><th style="width:120px;">Severity & Level</th><th style="width:180px;">Source & Destination</th><th>Evidence & Model Findings</th><th style="width:220px;">Recommended Action</th><th style="width:120px;">Firewall Status</th></tr></thead>
            <tbody>{threat_rows_html}</tbody>
        </table>
        <h2>2. Firewall Action & Response History</h2>
        <table>
            <thead><tr><th style="width:140px;">Timestamp</th><th style="width:90px;">Action</th><th style="width:160px;">Target IP</th><th>Execution Mode</th><th>Status Result</th></tr></thead>
            <tbody>{fw_rows_html}</tbody>
        </table>
        <h2>3. MITRE ATT&CK Contextual Framework Reference</h2>
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px; font-size:13px;">
            <p><strong>Separation of Concerns:</strong> XGBoost, Autoencoder, Fusion Layer, and Dynamic Temporal GNN compute intrinsic anomaly and graph edge risks. The MITRE ATT&CK engine deterministically maps corroborated evidence to adversary tactics and techniques.</p>
        </div>
    </div>
</body>
</html>"""
    return Response(report_html, mimetype="text/html")
