import json
import os


class ThreatIntelligence:

    def __init__(self):

        self.blacklist = {}

        path = "datasets/threat_intelligence.json"

        if os.path.exists(path):

            with open(path, "r") as f:
                self.blacklist = json.load(f)

    def lookup(self, ip):

        if ip in self.blacklist:

            return self.blacklist[ip]

        return {
            "status": "Unknown",
            "category": "None",
            "confidence": "Low"
        }