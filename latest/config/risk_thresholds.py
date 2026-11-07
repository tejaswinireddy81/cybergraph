"""
CyberGraph central risk threshold configuration.

All risk-related thresholds should reference this module to ensure
consistency across the pipeline.
"""

# Top-level flow risk alert tiers
# Risk < 0.35 -> Level 1 - Normal
# 0.35 <= Risk < 0.75 -> Level 2 - Suspicious
# Risk >= 0.75 -> Level 3 - High Risk
RISK_L1_MAX = 0.35
RISK_L2_MAX = 0.75

# BARI contextual thresholds
# BARI < 0.35 -> benign baseline
# 0.35 <= BARI < 0.65 -> progression / deviation concern
# BARI >= 0.65 -> contributes to Level 3 classification
BARI_L1_MAX = 0.35
BARI_L2_MAX = 0.65

# XAI / model signal thresholds
XGB_PROB_LOW = 0.30
XGB_PROB_HIGH = 0.75
AE_SCORE_HIGH = 0.5
SHAP_PRIMARY_CUTOFF = 0.50

# Correlation threshold
CORRELATION_THRESHOLD = 0.40

# Descriptive GNN risk labels
GNN_RISK_LOW = 0.20
GNN_RISK_MEDIUM = 0.35
GNN_RISK_HIGH = 0.65

# Attack journey / threat priority coefficients
TPS_WEIGHT_RISK = 0.35
TPS_WEIGHT_SENSITIVITY = 0.25
TPS_WEIGHT_PROGRESSION = 0.20
TPS_WEIGHT_PERSISTENCE = 0.20


def alert_tier(risk_score: float) -> str:
    """Determine the top-level alert tier from a risk score."""
    if risk_score < RISK_L1_MAX:
        return "LEVEL1_NORMAL"
    if risk_score < RISK_L2_MAX:
        return "LEVEL2_SUSPICIOUS"
    return "LEVEL3_CRITICAL"


def bari_tier(bari_score: float) -> str:
    """Determine the BARI-influenced alert tier."""
    if bari_score < BARI_L1_MAX:
        return "BARI_L1"
    if bari_score < BARI_L2_MAX:
        return "BARI_L2"
    return "BARI_L3"