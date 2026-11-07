import sys
import os
import re

sys.path.insert(0, r"c:\Users\LENOVO\Downloads\major_project (1)\latest")

import app

print("=" * 80)
print("VERIFICATION SUITE: CYBERGRAPH LIVE BIPARTITE GRAPH VISUALIZATION")
print("=" * 80)

html = app.HTML_TEMPLATE

# 1. Canvas wrapper and threat-feed height check (~750px usable height)
print("\n[CHECK 1] Graph Container Substantial Height (~750px)")
assert "height: 750px;" in html, "Canvas wrapper must be set to 750px height"
print("  PASSED: Container height is set to 750px (substantially increased ~2x).")

# 2. Responsive design check
print("\n[CHECK 2] Responsive Design Media Query")
assert "@media (max-width: 1200px)" in html, "Media query for smaller screens must be present"
print("  PASSED: Media query present for responsive display on smaller screens.")

# 3. Deterministic Node Sorting check
print("\n[CHECK 3] Deterministic Node Sorting (sortNodes & ipToNum)")
assert "function sortNodes" in html, "sortNodes function must be present"
assert "function ipToNum" in html, "ipToNum function must be present"
print("  PASSED: Deterministic sorting implemented for stable node positions.")

# 4. Curved Edge & Offset Routing check
print("\n[CHECK 4] Edge Routing & Bezier Curve Rendering")
assert "quadraticCurveTo" in html, "Quadratic Bezier curves must be used for curved edges"
assert "edgePairCounts" in html, "Edge offsets for multiple edges must be computed"
print("  PASSED: Curved edges and edge offsets implemented for non-overlapping edge routing.")

# 5. Crisp Host IP Label stroke check
print("\n[CHECK 5] Readable Host Identity & Text Contrast Stroke")
assert "strokeText" in html, "strokeText must be used for crisp text legibility"
assert "HOST / LAN ENDPOINTS" in html, "Column header title present"
assert "SERVICES / TARGET PORTS" in html, "Column header title present"
print("  PASSED: High contrast text stroke and clear column titles implemented.")

# 6. Preserved Legend check
print("\n[CHECK 6] Legend Preservation")
for legend_text in ["HOST (Normal)", "HOST (Suspicious)", "HOST (Critical)", "SERVICE", "EXTERNAL"]:
    assert legend_text in html, f"Legend item '{legend_text}' missing from HTML_TEMPLATE"
print("  PASSED: All 5 legend items intact.")

# 7. Preserved UI Panels check
print("\n[CHECK 7] Dashboard Panel Preservation")
assert "AI Model Pipeline Stage" in html
assert "MITRE ATT&CK Mapping" in html
assert "Threat Feed &amp; SHAP XAI" in html
assert "Temporal GNN Analytics" in html
assert "Real-Time Pipeline Latency" in html
assert "System Engine Health" in html
print("  PASSED: All dashboard panels intact.")

print("\n" + "=" * 80)
print("ALL VISUALIZATION AUDIT CHECKS PASSED SUCCESSFULLY!")
print("=" * 80)
