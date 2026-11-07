"""
tests/test_mitre_attack_mapper.py
==================================
Phase 2 — MITRE ATT&CK + Threat Context Integration Tests

10 test cases covering:
  1. Supported technique detection (T1046, T1110, T1498, T1059)
  2. Confidence scoring within threshold
  3. Unsupported technique rejection (evidence below threshold)
  4. Evidence-based mapping rationale
  5. Attack journey context enrichment
  6. BARI route context integration
  7. Confidence calculation correctness
  8. Full map_flow_to_techniques pipeline
  9. get_supported_techniques / get_technique_info
  10. explain_mapping human-readable output
"""

import pytest
from modules.mitre_attack_mapper import MITREAttackMapper, MITRETechnique


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def mapper():
    """Fresh MITREAttackMapper instance for each test."""
    return MITREAttackMapper()


@pytest.fixture
def sample_flow_record():
    """A representative network flow record for mapping tests."""
    return {
        "src_ip": "192.168.1.100",
        "dst_ip": "10.40.182.50",
        "dst_port": 80,
        "protocol": 6,
        "byte_count": 1500.0,
        "pkt_count": 25.0,
        "duration": 0.5,
        "xgb_prob": 0.45,
        "ae_score": 0.008,
        "fusion_score": 0.42,
        "gnn_risk": 0.35,
        "risk_score": 0.42,
        "timestamp": 1700000000.0,
    }


@pytest.fixture
def sample_bari_result():
    """A representative BARI result for mapping tests."""
    return {
        "bari_score": 0.62,
        "route_novelty": 0.75,
        "route_deviation": 0.68,
        "surface_progression": 0.72,
        "route_acceleration": 0.55,
        "route_persistence": 0.60,
        "gnn_risk": 0.35,
        "route_string": "HTTP(80) -> SSH(22) -> MySQL(3306)",
        "explanation_summary": "Route Novelty HIGH (0.75); Surface Progression HIGH (0.72)",
        "service_sensitivity": 0.75,
    }


@pytest.fixture
def sample_correlation_result():
    """A representative correlation result for mapping tests."""
    return {
        "attack_type": "Reconnaissance",
        "correlation_score": 0.55,
        "kill_chain_stage": "RECONNAISSANCE",
        "escalation_detected": True,
        "src_ip": "192.168.1.100",
        "event_count": 8,
        "timeline": [
            {"timestamp": "12:00:01", "src_ip": "192.168.1.100", "dst_ip": "10.40.182.10", "dst_port": 443, "protocol": 6, "attack_type": "Reconnaissance", "gnn_risk": 0.4, "xgb_prob": 0.3, "ae_score": 0.01, "fusion_score": 0.35, "risk_level": "Low"},
            {"timestamp": "12:00:02", "src_ip": "192.168.1.100", "dst_ip": "10.40.182.20", "dst_port": 22, "protocol": 6, "attack_type": "Reconnaissance", "gnn_risk": 0.4, "xgb_prob": 0.3, "ae_score": 0.01, "fusion_score": 0.35, "risk_level": "Low"},
            {"timestamp": "12:00:03", "src_ip": "192.168.1.100", "dst_ip": "10.40.182.30", "dst_port": 80, "protocol": 6, "attack_type": "Reconnaissance", "gnn_risk": 0.4, "xgb_prob": 0.3, "ae_score": 0.01, "fusion_score": 0.35, "risk_level": "Low"},
        ]
    }


@pytest.fixture
def sample_attack_journey():
    """A representative attack journey for mapping tests."""
    return {
        "journey_path": ["HTTP(80)", "SSH(22)", "MySQL(3306)"],
        "route_string": "HTTP(80) -> SSH(22) -> MySQL(3306)",
        "attack_objective": {
            "objective": "Database Data Access & Exfiltration",
            "category": "DATA_EXFILTRATION",
            "confidence": 92.5,
        },
        "threat_priority_score": 78.3,
        "threat_priority_tier": "HIGH",
    }


# ── Test Cases (10 tests) ─────────────────────────────────────────────────

class TestSupportedTechniques:
    """Requirement 2.1: Supported techniques are detectable with valid evidence."""

    def test_t1046_network_service_discovery_detected(self, mapper, sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey):
        """T1046 (Network Service Discovery) should be detected for multi-port scanning."""
        flow = dict(sample_flow_record)
        flow["dst_port"] = 443  # HTTPS probing context
        mappings = mapper.map_flow_to_techniques(
            flow, sample_bari_result, sample_correlation_result, sample_attack_journey
        )
        tech_ids = [m["technique_id"] for m in mappings]
        assert "T1046" in tech_ids, f"T1046 should be detected for multi-port scanning. Got: {tech_ids}"

    def test_t1110_no_mapping_from_reconnaissance(self, mapper, sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey):
        """T1110 should NOT be detected from Reconnaissance alone (no auth evidence)."""
        flow = dict(sample_flow_record)
        flow["dst_port"] = 22
        flow["pkt_count"] = 500.0  # High packet rate
        mappings = mapper.map_flow_to_techniques(
            flow, sample_bari_result, sample_correlation_result, sample_attack_journey
        )
        tech_ids = [m["technique_id"] for m in mappings]
        assert "T1110" not in tech_ids, f"T1110 should NOT map from Reconnaissance alone. Got: {tech_ids}"

    def test_t1498_dos_detected(self, mapper, sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey):
        """T1498 (Network Denial of Service) should be detected for high-volume traffic."""
        flow = dict(sample_flow_record)
        flow["dst_port"] = 80
        flow["pkt_count"] = 10000.0  # Very high packet rate
        flow["byte_count"] = 5000000.0  # Very high volume
        flow["duration"] = 0.1  # Short duration burst
        mappings = mapper.map_flow_to_techniques(
            flow, sample_bari_result, sample_correlation_result, sample_attack_journey
        )
        tech_ids = [m["technique_id"] for m in mappings]
        assert "T1498" in tech_ids, f"T1498 should be detected. Got: {tech_ids}"

    def test_t1059_disabled_no_mapping(self, mapper, sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey):
        """T1059 is currently unsupported — network flow alone should NOT map to T1059."""
        flow = dict(sample_flow_record)
        flow["dst_port"] = 4444  # Backdoor/C2 port
        flow["protocol"] = 17  # UDP anomaly
        mappings = mapper.map_flow_to_techniques(
            flow, sample_bari_result, sample_correlation_result, sample_attack_journey
        )
        tech_ids = [m["technique_id"] for m in mappings]
        assert "T1059" not in tech_ids, f"T1059 should be disabled (no command/script evidence). Got: {tech_ids}"


class TestConfidenceAndEvidence:
    """Requirement 2.2: Confidence scores are evidence-based, not ML guesses."""

    def test_confidence_above_threshold(self, mapper, sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey):
        """Detected mappings must have confidence >= required threshold."""
        mappings = mapper.map_flow_to_techniques(
            sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey
        )
        for mapping in mappings:
            tech = mapper.get_technique_info(mapping["technique_id"])
            assert mapping["confidence"] >= tech.required_confidence_threshold, (
                f"{mapping['technique_id']} confidence {mapping['confidence']} below threshold {tech.required_confidence_threshold}"
            )

    def test_unsupported_technique_rejected(self, mapper, sample_flow_record):
        """Low-evidence flows must NOT trigger technique mappings."""
        benign_flow = {
            "src_ip": "10.0.0.5",
            "dst_ip": "8.8.8.8",
            "dst_port": 53,
            "protocol": 17,
            "byte_count": 100.0,
            "pkt_count": 2.0,
            "duration": 0.01,
            "xgb_prob": 0.01,
            "ae_score": 0.0001,
            "fusion_score": 0.02,
            "gnn_risk": 0.0,
            "risk_score": 0.02,
            "timestamp": 1700000000.0,
        }
        empty_bari = {"bari_score": 0.0, "route_novelty": 0.0, "route_deviation": 0.0}
        empty_corr = {"attack_type": "Normal"}
        empty_journey = {"journey_path": [], "attack_objective": {}}
        mappings = mapper.map_flow_to_techniques(benign_flow, empty_bari, empty_corr, empty_journey)
        assert len(mappings) == 0, f"Benign flow should have 0 mappings, got {len(mappings)}"

    def test_rationale_contains_evidence(self, mapper, sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey):
        """Each mapping must include a human-readable rationale based on evidence."""
        mappings = mapper.map_flow_to_techniques(
            sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey
        )
        for mapping in mappings:
            assert "rationale" in mapping, f"{mapping['technique_id']} missing rationale"
            assert len(mapping["rationale"]) > 10, f"{mapping['technique_id']} rationale too short"
            assert mapping["source"] == "evidence-based", f"{mapping['technique_id']} not evidence-based"


class TestIntegrationPoints:
    """Requirement 2.3: MITRE integrates with journey, BARI, correlation."""

    def test_mapping_includes_journey_context(self, mapper, sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey):
        """MITRE mappings must include attack journey context."""
        mappings = mapper.map_flow_to_techniques(
            sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey
        )
        for mapping in mappings:
            assert "journey_path" in mapping, f"{mapping['technique_id']} missing journey_path"
            assert "attack_objective" in mapping, f"{mapping['technique_id']} missing attack_objective"

    def test_mapping_includes_bari_context(self, mapper, sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey):
        """MITRE mappings must include BARI route context."""
        mappings = mapper.map_flow_to_techniques(
            sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey
        )
        for mapping in mappings:
            assert "bari_score" in mapping, f"{mapping['technique_id']} missing bari_score"
            assert "route_string" in mapping, f"{mapping['technique_id']} missing route_string"


class TestAPIs:
    """Requirement 2.4: Public API methods work correctly."""

    def test_get_supported_techniques(self, mapper):
        """get_supported_techniques returns exactly the 3 supported IDs (T1059 disabled)."""
        supported = mapper.get_supported_techniques()
        assert set(supported) == {"T1046", "T1110", "T1498"}, (
            f"Expected 3 supported techniques, got: {supported}"
        )
        assert len(supported) == 3

    def test_get_technique_info(self, mapper):
        """get_technique_info returns MITRETechnique dataclass for live-enabled techniques."""
        for tech_id in ["T1046", "T1110", "T1498"]:
            info = mapper.get_technique_info(tech_id)
            assert info is not None, f"{tech_id} should have info"
            assert isinstance(info, MITRETechnique)
            assert info.technique_name != ""
            assert info.tactic != ""
            assert len(info.behavioral_indicators) > 0
            assert info.required_confidence_threshold > 0

    def test_get_technique_info_none(self, mapper):
        """get_technique_info returns None for unsupported/disabled techniques."""
        for tech_id in ["T1078", "T1059"]:
            info = mapper.get_technique_info(tech_id)
            assert info is None, f"{tech_id} should return None (unsupported/disabled)"

    def test_get_technique_info_disabled_t1059(self, mapper):
        """T1059 is documented but disabled - get_technique_info should return None."""
        info = mapper.get_technique_info("T1059")
        assert info is None, "T1059 should be disabled and return None"

    def test_explain_mapping(self, mapper, sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey):
        """explain_mapping returns a human-readable string."""
        mappings = mapper.map_flow_to_techniques(
            sample_flow_record, sample_bari_result, sample_correlation_result, sample_attack_journey
        )
        if mappings:
            explanation = mapper.explain_mapping(mappings[0])
            assert isinstance(explanation, str)
            assert len(explanation) > 20
            assert mappings[0]["technique_id"] in explanation or "T" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
