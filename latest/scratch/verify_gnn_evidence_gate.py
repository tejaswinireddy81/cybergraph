"""
verify_gnn_evidence_gate.py
===========================
Regression test for the tightened GNN evidence gate introduced to fix the
gateway ICMP false-positive.

Verified invariants
-------------------
Case 1 — Gateway ICMP false-positive:
    xgb=0.2351, ae=1.0, fusion=0.0145, gnn=0.9999
    Expected: base_ml_evidence=False, gnn_activation_reason=NONE,
              combined_risk=fusion=0.0145  (GNN does NOT dominate)

Case 2 — Legitimate strong-XGB detection:
    xgb=0.80, ae=1.0, fusion=0.20, gnn=0.99
    Expected: base_ml_evidence=True, gnn_activation_reason=XGB_STRONG,
              combined_risk=max(0.20, 0.99)=0.99  (GNN correctly participates)

Case 3 — AE anomaly with moderate XGB corroboration:
    xgb=0.45, ae=0.80, fusion=0.18, gnn=0.95
    Expected: base_ml_evidence=True, gnn_activation_reason=AE_PLUS_CORROBORATION,
              combined_risk=max(0.18, 0.95)=0.95

Case 4 — AE anomaly alone (no XGB/Fusion corroboration):
    xgb=0.30, ae=0.90, fusion=0.10, gnn=0.88
    Expected: base_ml_evidence=False, gnn_activation_reason=NONE,
              combined_risk=0.10

Case 5 — Fusion-strong path:
    xgb=0.20, ae=0.30, fusion=0.50, gnn=0.80
    Expected: base_ml_evidence=True, gnn_activation_reason=FUSION_STRONG,
              combined_risk=max(0.50, 0.80)=0.80
"""

import sys
import os
import math

# ── Replicate the exact evidence gate logic from app.py ──────────────────────

def compute_gnn_evidence(xgb_prob: float, ae_score: float, base_fusion: float,
                          gnn_edge_risk: float):
    """
    Mirrors the guarded GNN evidence gate implemented in app.py.

    Returns:
        base_ml_evidence   : bool
        gnn_activation_reason : str  (XGB_STRONG | FUSION_STRONG | AE_PLUS_CORROBORATION | NONE)
        combined_risk      : float
    """
    if xgb_prob >= 0.70:
        gnn_activation_reason = "XGB_STRONG"
        base_ml_evidence = True
    elif base_fusion >= 0.35:
        gnn_activation_reason = "FUSION_STRONG"
        base_ml_evidence = True
    elif ae_score >= 0.50 and (xgb_prob >= 0.40 or base_fusion >= 0.20):
        gnn_activation_reason = "AE_PLUS_CORROBORATION"
        base_ml_evidence = True
    else:
        gnn_activation_reason = "NONE"
        base_ml_evidence = False

    if base_ml_evidence:
        combined_risk = max(base_fusion, gnn_edge_risk)
    else:
        combined_risk = base_fusion

    return base_ml_evidence, gnn_activation_reason, combined_risk


# ── Test harness ─────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def approx_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, abs_tol=tol)


def run_case(name, xgb, ae, fusion, gnn,
             expect_evidence: bool,
             expect_reason: str,
             expect_combined: float):
    evidence, reason, combined = compute_gnn_evidence(xgb, ae, fusion, gnn)

    ok_evidence = (evidence == expect_evidence)
    ok_reason   = (reason == expect_reason)
    ok_combined = approx_eq(combined, expect_combined)

    status = PASS if (ok_evidence and ok_reason and ok_combined) else FAIL
    print(f"[{status}] {name}")

    if not ok_evidence:
        print(f"        base_ml_evidence: got={evidence}  expected={expect_evidence}")
    if not ok_reason:
        print(f"        gnn_activation_reason: got={reason!r}  expected={expect_reason!r}")
    if not ok_combined:
        print(f"        combined_risk: got={combined:.6f}  expected={expect_combined:.6f}")

    return ok_evidence and ok_reason and ok_combined


results = []

# Case 1: Gateway ICMP false-positive (the investigated incident)
results.append(run_case(
    name="Case 1 — Gateway ICMP false-positive (xgb=0.2351, ae=1.0, fusion=0.0145, gnn=0.9999)",
    xgb=0.2351, ae=1.0, fusion=0.0145, gnn=0.9999,
    expect_evidence=False,
    expect_reason="NONE",
    expect_combined=0.0145,
))

# Case 2: Legitimate strong-XGB detection
results.append(run_case(
    name="Case 2 — Strong XGB (xgb=0.80, ae=1.0, fusion=0.20, gnn=0.99)",
    xgb=0.80, ae=1.0, fusion=0.20, gnn=0.99,
    expect_evidence=True,
    expect_reason="XGB_STRONG",
    expect_combined=0.99,
))

# Case 3: AE anomaly + moderate XGB corroboration
results.append(run_case(
    name="Case 3 — AE+XGB corroboration (xgb=0.45, ae=0.80, fusion=0.18, gnn=0.95)",
    xgb=0.45, ae=0.80, fusion=0.18, gnn=0.95,
    expect_evidence=True,
    expect_reason="AE_PLUS_CORROBORATION",
    expect_combined=0.95,
))

# Case 4: AE anomaly alone — no XGB/Fusion corroboration
results.append(run_case(
    name="Case 4 — AE anomaly alone, no corroboration (xgb=0.30, ae=0.90, fusion=0.10, gnn=0.88)",
    xgb=0.30, ae=0.90, fusion=0.10, gnn=0.88,
    expect_evidence=False,
    expect_reason="NONE",
    expect_combined=0.10,
))

# Case 5: Fusion-strong path
results.append(run_case(
    name="Case 5 — Fusion-strong (xgb=0.20, ae=0.30, fusion=0.50, gnn=0.80)",
    xgb=0.20, ae=0.30, fusion=0.50, gnn=0.80,
    expect_evidence=True,
    expect_reason="FUSION_STRONG",
    expect_combined=0.80,
))

# ── Summary ──────────────────────────────────────────────────────────────────
print()
passed = sum(results)
total  = len(results)
if passed == total:
    print(f"ALL {total} CASES PASSED -- OK")
    sys.exit(0)
else:
    print(f"{passed}/{total} CASES PASSED -- {total - passed} FAILED")
    sys.exit(1)
