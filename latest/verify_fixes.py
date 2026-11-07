"""
verify_fixes.py — Quick verification of all 4 bug fixes (A, B, C, D)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name} — {detail}")
        failed += 1

print("=" * 60)
print("CYBERGRAPH BUG FIX VERIFICATION")
print("=" * 60)

# ================================================================
# FIX D: TTL domain shift — FEATURE_COLS should be 47, no TTL
# ================================================================
print("\n--- FIX D: TTL Domain Shift Elimination ---")

from train_xgboost_ae import FEATURE_COLS as XGB_COLS
check("train_xgboost_ae.py has 47 features", len(XGB_COLS) == 47, f"got {len(XGB_COLS)}")
check("train_xgboost_ae.py no MIN_TTL", "MIN_TTL" not in XGB_COLS)
check("train_xgboost_ae.py no MAX_TTL", "MAX_TTL" not in XGB_COLS)

# train_models.py — can't import (runs training at module level), so parse
tm_content = open("train_models.py").read()
# Check no quoted "MIN_TTL" appears as a list item (not in a comment)
tm_lines = [l.strip() for l in tm_content.split('\n') if '"MIN_TTL"' in l and not l.strip().startswith('#')]
check("train_models.py MIN_TTL only in comments", len(tm_lines) == 0, f"found: {tm_lines}")

# train_fusion.py
tf_content = open("train_fusion.py").read()
tf_lines = [l.strip() for l in tf_content.split('\n') if '"MIN_TTL"' in l and not l.strip().startswith('#')]
check("train_fusion.py MIN_TTL only in comments", len(tf_lines) == 0, f"found: {tf_lines}")

# flow_extractor.py
fe_content = open("flow_extractor.py").read()
# MIN_TTL should not appear as a dict key (feature export)
fe_key_lines = [l.strip() for l in fe_content.split('\n') if '"MIN_TTL"' in l and ':' in l and 'float(' in l]
check("flow_extractor.py no MIN_TTL export", len(fe_key_lines) == 0, f"found: {fe_key_lines}")

# ================================================================
# FIX A/B: Metadata Routing — _src_ip/_dst_ip prioritized
# ================================================================
print("\n--- FIX A/B: Metadata Routing ---")

lte_content = open("modules/live_traffic_engine.py").read()
check("live_traffic_engine.py uses _src_ip", '_src_ip' in lte_content)
check("live_traffic_engine.py no L4_SRC_PORT fallback for IP",
      'L4_SRC_PORT' not in lte_content.split('src_ip =')[1].split('\n')[0])

fsv_content = open("modules/feature_schema_validator.py").read()
check("feature_schema_validator.py uses _src_ip", '_src_ip' in fsv_content.split('extract_operational_fields')[1])
check("feature_schema_validator.py no L4_SRC_PORT as IP fallback",
      'L4_SRC_PORT' not in fsv_content.split('src_ip =')[1].split('\n')[0])

# ================================================================
# FIX C: Strict Schema Validation
# ================================================================
print("\n--- FIX C: Strict Schema Validation ---")

check("validator tracks errors", 'errors.append(f"Missing feature:' in fsv_content)
check("live_traffic_engine raises ValueError", 'raise ValueError' in lte_content)

# ================================================================
# Functional test: validate_schema detects missing features
# ================================================================
print("\n--- Functional Test: Schema Validation ---")

from modules.feature_schema_validator import FeatureSchemaValidator
v = FeatureSchemaValidator()

# Test with empty dict (all features missing)
is_valid, errors, arr = v.validate_schema({})
check("Empty dict detected as invalid", is_valid == False and len(errors) == 47,
      f"is_valid={is_valid}, errors_count={len(errors)}")

# Test with complete feature set
complete = {col: 1.0 for col in XGB_COLS}
is_valid2, errors2, arr2 = v.validate_schema(complete)
check("Complete dict detected as valid", is_valid2 == True and len(errors2) == 0,
      f"is_valid={is_valid2}, errors_count={len(errors2)}")
check("Feature vector length is 47", len(arr2) == 47, f"got {len(arr2)}")

# Test extract_operational_fields with FlowExtractor-style keys
test_flow = {"_src_ip": "192.168.1.100", "_dst_ip": "10.0.0.1", "_dst_port": 443, "PROTOCOL": 6}
op = v.extract_operational_fields(test_flow, model_risk=0.5)
check("Operational fields use _src_ip", op["source_host"] == "HOST:192.168.1.100",
      f"got {op['source_host']}")
check("Operational fields use _dst_ip", op["destination_host"] == "HOST:10.0.0.1",
      f"got {op['destination_host']}")
check("Operational fields use _dst_port", op["service_port"] == "SERVICE:443/6",
      f"got {op['service_port']}")

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} checks")
print("=" * 60)

if failed > 0:
    sys.exit(1)
else:
    print("ALL FIXES VERIFIED SUCCESSFULLY!")
    sys.exit(0)
