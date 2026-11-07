"""
modules/attack_journey_engine.py
=================================
CyberGraph Attack Journey Intelligence Engine

Elevates threat detection from isolated flow alerts into full "Attack Journey Intelligence":
  1. Attack Journey Reconstruction (Host -> Web -> App -> DB -> Admin)
  2. Attack Objective Inference (AOI) with confidence scoring
  3. Next-Route Forecasting (NRF) using BARI Markov memory P(S_{t+1} | S_t)
  4. Threat Priority Score (TPS) for SOC operational prioritization
  5. Additive Explainable Risk Breakdown (+0.31 GNN, +0.21 RD, +0.18 ASP...)
  6. Natural Language "Attack Story" Generator
  7. Attack Behavioral Fingerprinting & Archetype Matching
  8. "What-If" Defensive Attack Path Simulator
"""

import time
import math
from collections import defaultdict, deque

PORT_NAMES = {
    80: "HTTP", 443: "HTTPS", 8080: "HTTP-Alt", 8443: "HTTPS-Alt", 8000: "Dev-Web",
    53: "DNS", 25: "SMTP", 110: "POP3", 143: "IMAP", 465: "SMTPS", 587: "Submission",
    21: "FTP", 23: "Telnet", 22: "SSH", 3306: "MySQL", 5432: "PostgreSQL", 1433: "MSSQL", 1521: "Oracle",
    3389: "RDP", 5900: "VNC", 445: "SMB", 139: "NetBIOS-SSN", 137: "NetBIOS-NS", 161: "SNMP",
    4444: "Backdoor/C2", 4445: "C2-Alt", 5555: "Android-ADB/C2", 6666: "IRC-Bot", 7777: "C2-Relay", 31337: "EliteBackdoor"
}


def _port_label(port: int) -> str:
    name = PORT_NAMES.get(int(port), "SVC")
    return f"{name}({port})"


class AttackJourneyEngine:
    """
    Attack Journey Intelligence Engine for CyberGraph.
    Tracks stateful attacker progression across temporal snapshots.
    """

    def __init__(self, max_history: int = 25):
        self.max_history = max_history
        # Per host state: { src_ip: { "route": [...], "timeline": [...], "early_t": float, "critical_t": float } }
        self._journeys = defaultdict(lambda: {
            "route": [],
            "timeline": [],
            "early_t": None,
            "critical_t": None,
            "first_t": None,
            "last_t": None,
        })

    def process_flow_journey(self, *, src_ip: str, dst_port: int, protocol: int = 6,
                             risk_score: float, bari_res: dict, level_str: str,
                             timestamp: float = None, bari_engine = None) -> dict:
        """
        Updates host attack journey state and returns complete Attack Journey Intelligence payload.
        """
        now = timestamp if timestamp is not None else time.time()
        if now > 1e11:  # Timestamp passed in milliseconds
            now = now / 1000.0
        try:
            time_str = time.strftime("%H:%M:%S", time.localtime(now))
        except Exception:
            time_str = time.strftime("%H:%M:%S", time.localtime(time.time()))

        port = int(dst_port)
        label = _port_label(port)

        state = self._journeys[src_ip]
        if state["first_t"] is None:
            state["first_t"] = now
        state["last_t"] = now

        # Record route label if distinct or new step
        if not state["route"] or state["route"][-1] != label:
            state["route"].append(label)
            if len(state["route"]) > self.max_history:
                state["route"].pop(0)

        # Track early detection vs critical timestamps
        if "Level 2" in level_str and state["early_t"] is None:
            state["early_t"] = now
        elif "Level 3" in level_str and state["critical_t"] is None:
            state["critical_t"] = now

        lead_time_sec = 0.0
        if state["early_t"] is not None and state["critical_t"] is not None:
            lead_time_sec = max(0.0, round(state["critical_t"] - state["early_t"], 1))
        elif state["early_t"] is not None:
            lead_time_sec = max(0.0, round(now - state["early_t"], 1))

        # Add to timeline
        timeline_entry = {
            "timestamp": now,
            "time_str": time_str,
            "service": label,
            "port": port,
            "level": level_str,
            "risk_score": round(float(risk_score), 4),
            "bari_score": float(bari_res.get("bari_score", 0.0)) if isinstance(bari_res, dict) else float(bari_res),
            "stage": self._map_threat_stage(port, bari_res),
        }
        state["timeline"].append(timeline_entry)
        if len(state["timeline"]) > self.max_history:
            state["timeline"].pop(0)

        # 1. Attack Objective Inference (AOI)
        curr_sens = float(bari_res.get("service_sensitivity", 0.50)) if isinstance(bari_res, dict) else 0.50
        aoi = self.infer_attack_objective(state["route"], port, curr_sens)

        # 2. Next-Route Forecasting (NRF)
        forecast = []
        if bari_engine is not None and hasattr(bari_engine, "predict_next_targets"):
            forecast = bari_engine.predict_next_targets(port, protocol=protocol, top_k=4)
        else:
            forecast = [
                {"port": 3306, "label": "MySQL(3306)", "probability": 0.45, "sensitivity": 0.75},
                {"port": 22, "label": "SSH(22)", "probability": 0.30, "sensitivity": 0.65},
                {"port": 3389, "label": "RDP(3389)", "probability": 0.15, "sensitivity": 0.85},
                {"port": 4444, "label": "Backdoor(4444)", "probability": 0.10, "sensitivity": 1.00},
            ]

        # 3. Threat Priority Score (TPS)
        asp = float(bari_res.get("surface_progression", 0.0)) if isinstance(bari_res, dict) else 0.0
        rp = float(bari_res.get("route_persistence", 0.0)) if isinstance(bari_res, dict) else 0.0
        tps = self.compute_threat_priority_score(risk_score, curr_sens, asp, rp)

        # 4. Explainable Risk Breakdown
        gnn_risk = float(bari_res.get("gnn_risk", risk_score)) if isinstance(bari_res, dict) else float(risk_score)
        breakdown = self.compute_explainable_risk_breakdown(bari_res, gnn_risk)

        first_t_val = state["first_t"]
        if first_t_val and first_t_val > 1e11:
            first_t_val = first_t_val / 1000.0
        try:
            first_t_str = time.strftime("%H:%M:%S", time.localtime(first_t_val)) if first_t_val else time_str
        except Exception:
            first_t_str = time_str

        story = self.generate_attack_story(
            src_ip=src_ip,
            dst_port=port,
            level_str=level_str,
            bari_res=bari_res,
            objective_info=aoi,
            lead_time_sec=lead_time_sec,
            route=state["route"],
            first_t_str=first_t_str
        )


        # 6. Attack Fingerprint
        fingerprint = self.compute_attack_fingerprint(bari_res, state["route"])

        return {
            "source_ip": src_ip,
            "current_service": label,
            "journey_path": list(state["route"]),
            "route_string": " -> ".join(state["route"]),
            "timeline": list(state["timeline"]),
            "early_detection_time": state["early_t"],
            "critical_detection_time": state["critical_t"],
            "detection_lead_time_sec": lead_time_sec,
            "detection_lead_time_str": f"{lead_time_sec:.1f}s" if lead_time_sec > 0 else "Immediate",
            "attack_objective": aoi,
            "next_route_forecast": forecast,
            "threat_priority_score": tps,
            "threat_priority_tier": "HIGH" if tps > 75 else ("MEDIUM" if tps > 45 else "LOW"),
            "explainable_risk_breakdown": breakdown,
            "attack_story": story,
            "attack_fingerprint": fingerprint,
        }

    def _map_threat_stage(self, port: int, bari_res: dict) -> str:
        p = int(port)
        if p in (80, 443, 53, 8080):
            return "Reconnaissance / First Contact"
        elif p in (22, 21, 23):
            return "Initial Access / Remote Auth"
        elif p in (3389, 445, 139):
            return "Lateral Movement / Escalation"
        elif p in (3306, 5432, 1433, 1521):
            return "Target Exfiltration / Database Access"
        elif p in (4444, 4445, 5555, 31337):
            return "C2 Persistence / Objective Achieved"
        return "Route Anomaly Transition"

    def infer_attack_objective(self, route: list, current_port: int, current_sens: float) -> dict:
        """
        Derives the attacker's ultimate probable objective and confidence.
        """
        port = int(current_port)
        if port in (3306, 5432, 1433, 1521):
            return {
                "objective": "Database Data Access & Exfiltration",
                "category": "DATA_EXFILTRATION",
                "confidence": 92.5,
                "target_service": _port_label(port),
                "severity": "CRITICAL"
            }
        elif port in (3389, 5900, 445, 139):
            return {
                "objective": "Domain Privilege Escalation & Lateral Movement",
                "category": "LATERAL_MOVEMENT",
                "confidence": 88.0,
                "target_service": _port_label(port),
                "severity": "HIGH"
            }
        elif port in (4444, 4445, 5555, 6666, 7777, 31337):
            return {
                "objective": "C2 Command & Control / Persistence Installation",
                "category": "COMMAND_AND_CONTROL",
                "confidence": 96.0,
                "target_service": _port_label(port),
                "severity": "CRITICAL"
            }
        elif port in (22, 21, 23):
            return {
                "objective": "Remote Shell Access & Credential Harvesting",
                "category": "CREDENTIAL_ACCESS",
                "confidence": 84.0,
                "target_service": _port_label(port),
                "severity": "HIGH"
            }
        elif port in (80, 443, 8080, 8443):
            if len(route) > 2:
                return {
                    "objective": "Web Application Exploitation & Pivot Preparation",
                    "category": "WEB_EXPLOITATION",
                    "confidence": 78.5,
                    "target_service": _port_label(port),
                    "severity": "MEDIUM"
                }
            return {
                "objective": "Reconnaissance & Vulnerability Discovery",
                "category": "RECONNAISSANCE",
                "confidence": 70.0,
                "target_service": _port_label(port),
                "severity": "LOW"
            }
        else:
            return {
                "objective": "Unclassified Service Exposure & Anomaly",
                "category": "UNKNOWN_ANOMALY",
                "confidence": 60.0,
                "target_service": _port_label(port),
                "severity": "MEDIUM"
            }

    def compute_threat_priority_score(self, risk_score: float, sensitivity: float,
                                     progression: float, persistence: float) -> float:
        """
        Calculates Threat Priority Score (TPS) in [0.0, 100.0].
        TPS = (0.35 * Risk + 0.25 * Sensitivity + 0.20 * Progression + 0.20 * Persistence) * 100
        """
        raw = (0.35 * float(risk_score) +
               0.25 * float(sensitivity) +
               0.20 * float(progression) +
               0.20 * float(persistence))
        return round(min(100.0, max(0.0, raw * 100.0)), 2)

    def compute_explainable_risk_breakdown(self, bari_res: dict, gnn_risk: float) -> dict:
        """
        Computes additive attribution scores for the final risk breakdown.
        """
        if not isinstance(bari_res, dict):
            bari_res = {}
        rn = float(bari_res.get("route_novelty", 0.0))
        rd = float(bari_res.get("route_deviation", 0.0))
        asp = float(bari_res.get("surface_progression", 0.0))
        ra = float(bari_res.get("route_acceleration", 0.0))
        rp = float(bari_res.get("route_persistence", 0.0))
        g = float(gnn_risk)

        # Weighted contributions (matching BARI fusion weights)
        c_gnn = round(0.20 * g, 4)
        c_rd  = round(0.15 * rd, 4)
        c_asp = round(0.20 * asp, 4)
        c_rn  = round(0.20 * rn, 4)
        c_rp  = round(0.10 * rp, 4)
        c_ra  = round(0.15 * ra, 4)

        total = round(c_gnn + c_rd + c_asp + c_rn + c_rp + c_ra, 4)

        return {
            "gnn_risk_contribution": c_gnn,
            "route_deviation_contribution": c_rd,
            "surface_progression_contribution": c_asp,
            "route_novelty_contribution": c_rn,
            "route_persistence_contribution": c_rp,
            "route_acceleration_contribution": c_ra,
            "total_risk_score": total,
            "breakdown_list": [
                {"component": "Temporal GNN Risk", "weight": "+0.20", "score": round(g, 2), "contribution": c_gnn},
                {"component": "Attack-Surface Progression", "weight": "+0.20", "score": round(asp, 2), "contribution": c_asp},
                {"component": "Route Novelty", "weight": "+0.20", "score": round(rn, 2), "contribution": c_rn},
                {"component": "Route Deviation", "weight": "+0.15", "score": round(rd, 2), "contribution": c_rd},
                {"component": "Route Acceleration", "weight": "+0.15", "score": round(ra, 2), "contribution": c_ra},
                {"component": "Route Persistence", "weight": "+0.10", "score": round(rp, 2), "contribution": c_rp},
            ]
        }

    def generate_attack_story(self, *, src_ip: str, dst_port: int, level_str: str,
                             bari_res: dict, objective_info: dict, lead_time_sec: float,
                             route: list, first_t_str: str) -> str:
        """
        Generates a human-readable natural language narrative detailing the attack journey.
        """
        route_str = " -> ".join(route) if route else _port_label(dst_port)
        obj = objective_info.get("objective", "Unauthorized Service Access")
        conf = objective_info.get("confidence", 80.0)

        lead_str = f"{lead_time_sec:.1f} seconds" if lead_time_sec > 0 else "early detection stage"

        if "Level 3" in level_str:
            return (
                f"ALERT: Host {src_ip} initiated abnormal service transitions beginning at {first_t_str}. "
                f"The host departed from its learned benign baseline and executed a temporal route progression ({route_str}). "
                f"CyberGraph identified this activity at Level 2 Early Warning {lead_str} prior to critical escalation. "
                f"Inferred probable objective: {obj} ({conf:.1f}% confidence). "
                f"Active firewall response executed to isolate host {src_ip}."
            )
        elif "Level 2" in level_str:
            return (
                f"EARLY WARNING: Host {src_ip} exhibited abnormal route behavior starting at {first_t_str}. "
                f"The host initiated transition along route ({route_str}), approaching sensitive target service {_port_label(dst_port)}. "
                f"Probable early objective: {obj} ({conf:.1f}% confidence). "
                f"Current status: Level 2 Early Warning active; monitoring route persistence before critical response."
            )
        else:
            return f"Host {src_ip} operating within normal route baseline ({route_str}). No threat detected."

    def compute_attack_fingerprint(self, bari_res: dict, route: list) -> dict:
        """
        Computes a 6-vector behavioral signature and matches against attack archetypes.
        """
        if not isinstance(bari_res, dict):
            bari_res = {}
        vec = {
            "route_novelty": float(bari_res.get("route_novelty", 0.0)),
            "route_deviation": float(bari_res.get("route_deviation", 0.0)),
            "surface_progression": float(bari_res.get("surface_progression", 0.0)),
            "route_acceleration": float(bari_res.get("route_acceleration", 0.0)),
            "route_persistence": float(bari_res.get("route_persistence", 0.0)),
            "gnn_risk": float(bari_res.get("gnn_risk", 0.0)),
        }

        # Archetype matching
        if vec["surface_progression"] > 0.70 and vec["route_deviation"] > 0.70:
            archetype = "Lateral Movement & Sensitive Target Escalation"
            archetype_id = "ARCH-01-LATERAL"
        elif vec["route_acceleration"] > 0.75 and vec["route_novelty"] > 0.70:
            archetype = "Automated Multi-Service Scan / Burst Probe"
            archetype_id = "ARCH-02-RECON_BURST"
        elif vec["route_persistence"] > 0.70 and vec["gnn_risk"] > 0.75:
            archetype = "Persistent Multi-Stage Advanced Intrusion"
            archetype_id = "ARCH-03-PERSISTENT_APT"
        else:
            archetype = "Emerging Anomaly Transition Pattern"
            archetype_id = "ARCH-04-EMERGING"

        return {
            "archetype": archetype,
            "archetype_id": archetype_id,
            "signature_vector": vec,
            "observed_route_length": len(route),
        }

    def simulate_attack_path(self, start_host: str = "10.0.0.21", target_port: int = 3306,
                            strategy: str = "lateral_movement", bari_engine = None) -> dict:
        """
        Defensive "What-If" Attack Path Simulator for SOC operators and research testing.
        Returns predicted attack journey, risk trajectory, detection stage, and lead time.
        """
        port = int(target_port)
        target_name = _port_label(port)

        if strategy == "lateral_movement":
            sim_path = ["HTTP(80)", "HTTPS(443)", "SSH(22)", target_name]
            sim_risks = [0.15, 0.38, 0.65, 0.91]
            sim_baris = [0.20, 0.42, 0.68, 0.88]
        elif strategy == "stealthy_exfiltration":
            sim_path = ["DNS(53)", "HTTPS(443)", "PostgreSQL(5432)", target_name]
            sim_risks = [0.08, 0.25, 0.55, 0.86]
            sim_baris = [0.15, 0.35, 0.60, 0.82]
        elif strategy == "direct_exploit":
            sim_path = ["HTTP(80)", target_name]
            sim_risks = [0.20, 0.95]
            sim_baris = [0.25, 0.92]
        else:
            sim_path = ["HTTP(80)", "HTTP-Alt(8080)", "MySQL(3306)", target_name]
            sim_risks = [0.12, 0.40, 0.72, 0.94]
            sim_baris = [0.18, 0.45, 0.75, 0.90]

        early_warn_step = None
        critical_step = None
        for idx, (r, b) in enumerate(zip(sim_risks, sim_baris)):
            if (b >= 0.35 or r >= 0.35) and early_warn_step is None:
                early_warn_step = idx + 1
            if (r >= 0.75 or b >= 0.65) and critical_step is None:
                critical_step = idx + 1

        lead_steps = (critical_step - early_warn_step) if (critical_step and early_warn_step) else 1
        predicted_lead_sec = lead_steps * 15.0

        obj_info = self.infer_attack_objective(sim_path, port, 0.80)

        forecast = []
        if bari_engine is not None and hasattr(bari_engine, "predict_next_targets"):
            forecast = bari_engine.predict_next_targets(port, top_k=3)
        else:
            forecast = [
                {"port": 3389, "label": "RDP(3389)", "probability": 0.52, "sensitivity": 0.85},
                {"port": 4444, "label": "Backdoor(4444)", "probability": 0.33, "sensitivity": 1.00},
                {"port": 445, "label": "SMB(445)", "probability": 0.15, "sensitivity": 0.85},
            ]

        return {
            "simulation_id": f"SIM-{int(time.time())}",
            "start_host": start_host,
            "target_service": target_name,
            "strategy": strategy,
            "simulated_journey_path": sim_path,
            "risk_trajectory": sim_risks,
            "bari_trajectory": sim_baris,
            "predicted_final_risk": sim_risks[-1],
            "predicted_threat_level": "Level 3 — Critical / Confirmed" if sim_risks[-1] >= 0.75 else "Level 2 — Suspicious",
            "early_warning_step": early_warn_step,
            "critical_escalation_step": critical_step,
            "predicted_lead_time_steps": lead_steps,
            "predicted_lead_time_sec": predicted_lead_sec,
            "predicted_objective": obj_info,
            "forecasted_next_targets": forecast,
            "simulation_summary": (
                f"Simulated attack path ({' -> '.join(sim_path)}) targeting {target_name}. "
                f"CyberGraph Early Warning predicted at Step {early_warn_step}, providing a "
                f"{predicted_lead_sec:.0f}s ({lead_steps} step) lead time advantage prior to Critical escalation."
            )
        }

    def get_top_threat_priorities(self, limit: int = 10) -> list:
        """
        Returns active attacker journeys ranked by Threat Priority Score (TPS).
        """
        priorities = []
        for src_ip, data in self._journeys.items():
            if not data["timeline"]:
                continue
            last_entry = data["timeline"][-1]
            port = last_entry.get("port", 80)
            sens_val = 0.85 if port in (3306, 5432, 3389, 4444, 445) else 0.40
            risk = last_entry["risk_score"]
            tps = self.compute_threat_priority_score(risk, sens_val, 0.70, 0.60)
            
            priorities.append({
                "source_ip": src_ip,
                "current_service": last_entry["service"],
                "journey_length": len(data["route"]),
                "route_string": " -> ".join(data["route"]),
                "last_risk_score": risk,
                "last_bari_score": last_entry["bari_score"],
                "threat_level": last_entry["level"],
                "threat_priority_score": tps,
                "threat_priority_tier": "HIGH" if tps > 75 else ("MEDIUM" if tps > 45 else "LOW"),
                "last_active": time.strftime("%H:%M:%S", time.localtime(last_entry["timestamp"])),
            })

        priorities.sort(key=lambda x: x["threat_priority_score"], reverse=True)
        return priorities[:limit]
