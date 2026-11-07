from collections import defaultdict


class AttackChainDetector:

    def __init__(self):

        self.history = defaultdict(list)

    def update(self, ip, attack):

        if attack == "Normal":
            return

        self.history[ip].append(attack)

        # Keep only last 10 events
        self.history[ip] = self.history[ip][-10:]

    def detect(self, ip):

        attacks = self.history[ip]

        chain = " -> ".join(attacks)

        if (
            "Port Scan" in attacks and
            "Brute Force" in attacks
        ):
            return "Reconnaissance Attack Chain"

        if (
            "Brute Force" in attacks and
            "Malware" in attacks
        ):
            return "Credential Compromise"

        if (
            "Malware" in attacks and
            "Data Exfiltration" in attacks
        ):
            return "Data Theft Attack"

        return chain