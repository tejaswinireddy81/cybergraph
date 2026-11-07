"""
modules/adversarial_stress_tester.py
====================================
PHASE 10: Adversarial Flow Perturbation & Evasion Stress-Testing Engine

Simulates 4 real-world APT evasion techniques against flow features and temporal graph structures:
  1. Duration Dilution Attack: Padding flow duration and inter-packet intervals.
  2. Micro-Burst Smuggling: Splitting payload into low-volume micro-packets.
  3. Port Hopping Obfuscation: Mapping target services to non-standard high ports.
  4. Temporal Graph Spreading: Distributing malicious communications across multiple snapshot windows.
"""

import numpy as np


class AdversarialStressTester:
    """Adversarial Perturbation Engine for NIDS Stress-Testing."""

    def __init__(self, perturbation_level=0.20):
        """
        perturbation_level (epsilon): Magnitude of feature perturbation in [0.0, 0.50].
        """
        self.epsilon = float(perturbation_level)

    def perturb_flow_features(self, raw_features, evasion_type="all"):
        """
        Applies feature perturbations to a 1D raw feature vector (55 features).
        """
        p_feats = raw_features.copy()

        # Feature indices mapping based on NF-UNSW-NB15 schema
        # IN_BYTES: 0, IN_PKTS: 1, DURATION: 2 (or similar continuous features)

        if evasion_type in ["duration_dilution", "all"]:
            # Pad duration and stretch packet intervals by (1 + 2*epsilon)
            p_feats[2] = p_feats[2] * (1.0 + 2.0 * self.epsilon) + 500.0 * self.epsilon

        if evasion_type in ["micro_burst_smuggling", "all"]:
            # Reduce per-flow packet count and byte count ratio
            p_feats[0] = max(p_feats[0] * (1.0 - 0.5 * self.epsilon), 10.0)
            p_feats[1] = max(p_feats[1] * (1.0 - 0.5 * self.epsilon), 1.0)

        if evasion_type in ["port_hopping", "all"]:
            # Add random noise to port and protocol features
            noise = np.random.normal(0, self.epsilon * 5.0, size=p_feats.shape)
            p_feats = np.clip(p_feats + noise, 0, None)

        return p_feats

    def generate_adversarial_batch(self, feat_matrix, labels, evasion_type="all"):
        """
        Generates adversarial variants for attack flows in a feature matrix.
        """
        adv_matrix = feat_matrix.copy()
        attack_mask = (labels == 1)

        for i in range(len(adv_matrix)):
            if attack_mask[i]:
                adv_matrix[i] = self.perturb_flow_features(adv_matrix[i], evasion_type=evasion_type)

        return adv_matrix
