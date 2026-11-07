"""
device_profile.py
=================

Maintains a live profile for every device seen on the network.

Each device stores:

- First Seen / Last Seen
- Total Flows / Packets / Bytes
- RF / AE / GAT / Fusion scores
- Risk Score / Risk Level
- Threat Intelligence / Attack Chain
- Current Status
- History
- Local vs external classification
- Incoming vs outgoing traffic, protocol distribution, top peers
  (from BehaviorAnalyzer -- see update()'s behavior_extra param)
- Attack count (how many updates were labelled "Attack" by RF)
- Most recent explanation (human-readable reasons + SHAP top features)
- Most recent response/incident info from the response policy engine

Designed for real-time dashboard visualization.
"""

from datetime import datetime
import json


class DeviceProfile:

    def __init__(self):
        self.devices = {}

    def update(
        self,
        ip,
        packets=0,
        bytes_count=0,
        rf_score=0.0,
        ae_score=0.0,
        gat_score=0.0,
        fusion_score=0.0,
        risk_score=0.0,
        risk_level="Low",
        threat="Unknown",
        attack_chain=None,
        status="Active",
        is_local=None,
        protocols=None,
        top_peers=None,
        in_packets=0,
        in_bytes=0,
        out_packets=0,
        out_bytes=0,
        explanation=None,
        response_action=None,
        incident_id=None,
    ):

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if ip not in self.devices:

            self.devices[ip] = {
                "ip": ip,
                "first_seen": now,
                "last_seen": now,
                "flows": 0,
                "packets": 0,
                "bytes": 0,
                "rf_score": 0,
                "ae_score": 0,
                "gat_score": 0,
                "fusion_score": 0,
                "risk_score": 0,
                "risk_level": "Low",
                "threat": "Unknown",
                "attack_chain": [],
                "attack_count": 0,
                "status": "Active",
                "history": [],
                "is_local": None,
                "protocols": {},
                "top_peers": [],
                "in_packets": 0,
                "in_bytes": 0,
                "out_packets": 0,
                "out_bytes": 0,
                "explanation": None,
                "response_action": None,
                "incident_id": None,
            }

        device = self.devices[ip]

        device["last_seen"] = now
        device["flows"] += 1
        device["packets"] += packets
        device["bytes"] += bytes_count

        device["rf_score"] = round(rf_score, 4)
        device["ae_score"] = round(ae_score, 4)
        device["gat_score"] = round(gat_score, 4)
        device["fusion_score"] = round(fusion_score, 4)

        device["risk_score"] = round(risk_score, 4)
        device["risk_level"] = risk_level

        device["threat"] = threat
        if threat == "Attack":
            device["attack_count"] += 1

        if attack_chain:
            device["attack_chain"] = attack_chain

        device["status"] = status

        if is_local is not None:
            device["is_local"] = is_local
        if protocols is not None:
            device["protocols"] = protocols
        if top_peers is not None:
            device["top_peers"] = top_peers
        device["in_packets"] = in_packets
        device["in_bytes"] = in_bytes
        device["out_packets"] = out_packets
        device["out_bytes"] = out_bytes

        if explanation is not None:
            device["explanation"] = explanation
        if response_action is not None:
            device["response_action"] = response_action
        if incident_id is not None:
            device["incident_id"] = incident_id

        device["history"].append({
            "time": now,
            "risk": risk_score,
            "level": risk_level
        })
        # keep history bounded -- this dict lives for the whole process
        # lifetime, so an unbounded list would leak memory on a long-running
        # capture session
        if len(device["history"]) > 200:
            device["history"] = device["history"][-200:]

        return device

    def get(self, ip):

        return self.devices.get(ip)

    def get_all(self):

        return self.devices

    def top_risky_devices(self, n=10):

        devices = sorted(
            self.devices.values(),
            key=lambda d: d["risk_score"],
            reverse=True
        )

        return devices[:n]

    def export_json(self, path):

        with open(path, "w") as f:
            json.dump(self.devices, f, indent=4, default=str)

    def clear(self):

        self.devices.clear()
