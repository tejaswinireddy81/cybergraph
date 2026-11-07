class RiskEngine:

    def calculate(
        self,
        rf_prob,
        ae_score,
        device_packets,
        unique_destinations,
        unique_ports
    ):

        risk = 0

        # Random Forest contribution
        risk += rf_prob * 40

        # Autoencoder contribution
        risk += ae_score * 30

        # High packet count
        if device_packets > 1000:
            risk += 10

        # Many destination IPs
        if unique_destinations > 20:
            risk += 10

        # Many ports contacted
        if unique_ports > 10:
            risk += 10

        risk = min(100, round(risk))

        if risk >= 80:
            level = "Critical"

        elif risk >= 60:
            level = "High"

        elif risk >= 40:
            level = "Medium"

        else:
            level = "Low"

        return risk, level