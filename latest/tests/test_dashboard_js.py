"""Guards against shipping a dashboard script that cannot run in the browser."""
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parent.parent / "static" / "js" / "app.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_has_valid_syntax():
    res = subprocess.run(["node", "--check", str(JS)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_no_duplicate_top_level_function_declarations():
    src = JS.read_text(encoding="utf-8")
    names = re.findall(r"^ {8}(?:async )?function ([A-Za-z_$][\w$]*)", src, re.M)
    dupes = [n for n, c in Counter(names).items() if c > 1]
    assert not dupes, f"duplicate function declarations: {dupes}"


def test_stream_handler_updates_graph_and_charts():
    src = JS.read_text(encoding="utf-8")
    assert "updateGraphLayout(data)" in src
    assert "updateAllCharts(data)" in src
    assert "renderFilteredAlerts()" in src
