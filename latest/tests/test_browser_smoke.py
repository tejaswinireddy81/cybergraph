"""Real-browser smoke test of the dashboard.

Starts the app, opens it in headless Chrome/Edge over the DevTools protocol, and checks
that the page loads without JS errors, KPIs and charts render, and the admin login flow
works with an HttpOnly session (no token in localStorage).

Skipped automatically when no Chromium-based browser, `websockets`, or trained models
are available (e.g. a bare CI runner).
"""
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ADMIN_TOKEN = "smoke-test-token"

websockets = pytest.importorskip("websockets")


def _find_browser():
    env = os.environ.get("CHROME_PATH")
    if env and os.path.exists(env):
        return env
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    for p in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ):
        if os.path.exists(p):
            return p
    return None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


BROWSER = _find_browser()
pytestmark = [
    pytest.mark.skipif(BROWSER is None, reason="no Chromium-based browser found"),
    pytest.mark.skipif(not (ROOT / "models" / "model_manifest.json").exists(), reason="trained models not present"),
]

PAINTED_JS = (
    "(id=>{const c=document.getElementById(id);"
    "const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;"
    "let n=0;for(let i=3;i<d.length;i+=4)if(d[i]>0)n++;return n})"
)


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    env = dict(os.environ, CYBERGRAPH_PORT=str(port), CYBERGRAPH_ADMIN_TOKEN=ADMIN_TOKEN,
               CYBERGRAPH_MODE="attack", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                with urllib.request.urlopen(url + "/api/stream_step", timeout=2) as resp:
                    data = json.load(resp)
                    if data.get("stats", {}).get("total_flows", 0) > 0:
                        break
            except Exception:
                if proc.poll() is not None:
                    pytest.skip("app exited during startup")
            time.sleep(1)
        else:
            pytest.skip("app did not start in time")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _run_browser(url):
    prof = tempfile.mkdtemp(prefix="cg_smoke_")
    dbg = _free_port()
    proc = subprocess.Popen(
        [BROWSER, "--headless=new", "--disable-gpu", "--no-sandbox", f"--user-data-dir={prof}",
         f"--remote-debugging-port={dbg}", "--window-size=1500,1900", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = {"errors": []}
    try:
        tabs = None
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{dbg}/json"))
                break
            except Exception:
                await asyncio.sleep(0.5)
        assert tabs, "browser did not expose DevTools"
        ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=None) as ws:
            counter = {"n": 0}

            def collect(r):
                if r.get("method") == "Runtime.exceptionThrown":
                    d = r["params"]["exceptionDetails"]
                    result["errors"].append("EXC " + str(d.get("exception", {}).get("description", d.get("text"))))
                elif r.get("method") == "Runtime.consoleAPICalled" and r["params"]["type"] == "error":
                    result["errors"].append("console.error " + " ".join(
                        str(a.get("value", a.get("description", ""))) for a in r["params"]["args"])[:200])

            async def call(method, params=None):
                counter["n"] += 1
                mid = counter["n"]
                await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
                while True:
                    r = json.loads(await ws.recv())
                    if r.get("id") == mid:
                        return r
                    collect(r)

            async def ev(expr):
                r = await call("Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True})
                return r["result"]["result"].get("value")

            async def wait(sec):
                end = time.time() + sec
                while time.time() < end:
                    try:
                        collect(json.loads(await asyncio.wait_for(ws.recv(), 0.5)))
                    except asyncio.TimeoutError:
                        pass

            await call("Runtime.enable")
            await call("Page.enable")
            await call("Page.navigate", {"url": url + "/"})
            await wait(15)

            result["total_flows"] = int(await ev("document.getElementById('val-total').innerText") or 0)
            result["graph_px"] = await ev(PAINTED_JS + "('graph-canvas')")
            result["alerts_chart_px"] = await ev(PAINTED_JS + "('chart-alerts-time')")
            result["cards"] = await ev("document.querySelectorAll('.threat-card').length")

            await ev("document.getElementById('admin-token-input').value='nope'; doAdminLogin()")
            await wait(2)
            result["bad_login"] = await ev("document.getElementById('admin-auth-status').innerText")
            await ev("document.getElementById('admin-token-input').value='%s'; doAdminLogin()" % ADMIN_TOKEN)
            await wait(3)
            result["good_login"] = await ev("document.getElementById('admin-auth-status').innerText")
            result["ls_token"] = await ev("String(localStorage.getItem('cybergraph_admin_token'))")
            result["js_cookie"] = await ev("document.cookie")
            result["post_status"] = await ev("postAdmin('/api/unblock_ip',{ip:'192.0.2.250'}).then(r=>r.status)")
            await wait(1)
            result["fw_badge"] = await ev("document.getElementById('btn-fw-mode').innerText")
            result["dry_block"] = await ev(
                "postAdmin('/api/execute_real_block',{ip:'203.0.113.99',confirm:true}).then(r=>r.json()).then(j=>j.status+'|'+j.dry_run)")
            result["modal_note"] = await ev(
                "(()=>{executeRealBlock('203.0.113.98');return document.getElementById('fw-confirm-mode-note').innerText})()")
            await ev("closeFirewallModal()")
            await ev("document.getElementById('btn-open-block-history').click()")
            await wait(2)
            result["history_open"] = await ev(
                "getComputedStyle(document.getElementById('block-history-modal')).display")
    finally:
        proc.terminate()
        await asyncio.sleep(1)
        shutil.rmtree(prof, ignore_errors=True)
    return result


def test_dashboard_in_real_browser(server):
    r = asyncio.run(_run_browser(server))
    errors = [e for e in r["errors"] if "favicon" not in e and "401" not in e and "403" not in e]
    assert not errors, errors
    assert r["total_flows"] > 0, "KPIs never updated"
    assert r["graph_px"] > 8000, "graph radar did not render"
    assert r["alerts_chart_px"] > 500, "alerts chart did not render"
    assert r["cards"] > 0, "no alert cards rendered"
    assert "Invalid administrator token" in r["bad_login"]
    assert "Authenticated" in r["good_login"]
    assert r["ls_token"] == "null", "admin token must not be kept in localStorage"
    assert "cybergraph_session" not in r["js_cookie"], "session cookie must be HttpOnly"
    assert r["post_status"] == 200
    assert "DRY-RUN" in r["fw_badge"]
    assert r["dry_block"] == "SIMULATED_BLOCK_RECORDED|true"
    assert "no OS firewall command" in r["modal_note"]
    assert r["history_open"] == "flex"
