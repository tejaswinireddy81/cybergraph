"""Shared pytest fixtures.

SAFETY: the test-suite must never modify the real OS firewall. Several tests exercise
/api/execute_real_block, which runs `netsh`/`iptables` (and a UAC-elevated PowerShell
fallback). This autouse fixture replaces subprocess.run for those commands with an
in-memory fake, so a test run cannot leave Block-GNN-* rules on the developer's machine.
Other subprocess calls pass through untouched.
"""
import re
import subprocess
from types import SimpleNamespace

import pytest

_REAL_RUN = subprocess.run
_FIREWALL_BINARIES = ("netsh", "iptables", "powershell")


def _rule_name(text):
    m = re.search(r'name="?([^"\s\']+)', text)
    return m.group(1) if m else None


@pytest.fixture(autouse=True)
def _fake_os_firewall(monkeypatch, tmp_path):
    rules = set()

    def fake_run(cmd, *args, **kwargs):
        if not (isinstance(cmd, (list, tuple)) and cmd and str(cmd[0]).lower() in _FIREWALL_BINARIES):
            return _REAL_RUN(cmd, *args, **kwargs)
        joined = " ".join(str(c) for c in cmd)
        name = _rule_name(joined)
        low = joined.lower()
        if "add rule" in low and name:
            rules.add(name)
            return SimpleNamespace(returncode=0, stdout="Ok.", stderr="")
        if "delete rule" in low and name:
            rules.discard(name)
            return SimpleNamespace(returncode=0, stdout="Deleted 1 rule(s).", stderr="")
        if "show rule" in low and name:
            if name in rules:
                return SimpleNamespace(returncode=0, stdout=f"Rule Name: {name}", stderr="")
            return SimpleNamespace(returncode=1, stdout="No rules match the specified criteria.", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    from modules.firewall_response_executor import FirewallResponseExecutor
    monkeypatch.setattr(FirewallResponseExecutor, "DEFAULT_AUDIT_LOG", str(tmp_path / "firewall_audit.jsonl"))
    yield rules


# ──────────────────────────────────────────────────────────────────────────────
# Fresh-checkout / CI support
#
# Trained models, the 88 MB dataset and local secrets are gitignored, so a fresh GitHub
# runner has none of them. Tests are therefore split explicitly:
#   * unmarked tests      -> web layer, firewall safety, auth, JS etc. They run everywhere.
#                            When models are absent they use a lightweight engine init that
#                            skips model/dataset loading, live capture and background threads.
#   * @requires_models    -> need real trained models; skipped (with a clear reason) when absent.
# On a developer machine WITH models nothing changes: everything runs against the real engines.
# ──────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_REQUIRED_ARTIFACTS = (
    _ROOT / "models" / "model_manifest.json",
    _ROOT / "models" / "xgboost_model.json",
    _ROOT / "models" / "scaler.pkl",
    _ROOT / "NF-UNSW-NB15-v3.csv",
)
MODELS_AVAILABLE = all(p.exists() for p in _REQUIRED_ARTIFACTS)


def pytest_collection_modifyitems(config, items):
    if MODELS_AVAILABLE:
        return
    skip = pytest.mark.skip(
        reason="trained models/dataset not present (run scripts/setup_pipeline.py to build them)"
    )
    for item in items:
        if "requires_models" in item.keywords:
            item.add_marker(skip)


def _light_init_engines():
    """Stand-in for app.init_engines(): web-layer engines only, no models / dataset / capture."""
    import app as a
    from modules.firewall_response_executor import FirewallResponseExecutor

    if a.fw_executor is not None:
        return
    a.fw_executor = FirewallResponseExecutor(mode="dry-run")
    a.traffic_engine = SimpleNamespace(light_test_stub=True)
    for attr, factory in (
        ("mitre_mapper", lambda: __import__("modules.mitre_attack_mapper", fromlist=["x"]).MITREAttackMapper()),
        ("correlation_engine", lambda: __import__("modules.correlation_engine", fromlist=["x"]).CorrelationEngine()),
        ("bari_engine", lambda: __import__("modules.bari_engine", fromlist=["x"]).BARIEngine()),
        ("attack_journey_engine", lambda: __import__("modules.attack_journey_engine", fromlist=["x"]).AttackJourneyEngine()),
    ):
        try:
            setattr(a, attr, factory())
        except Exception:  # an engine that cannot be built without models stays None
            setattr(a, attr, None)
    a.alert_engine = SimpleNamespace(light_test_stub=True, fw_executor=a.fw_executor, enable_real_blocking=False)
    a.pipeline_thread_started = True   # never start the background pipeline thread in tests
    a._capture_initialized = False


# Patch at conftest import time: several test modules do `from app import init_engines`,
# which binds the function before any fixture could replace it.
if not MODELS_AVAILABLE:
    import app as _app_module

    _app_module.init_engines = _light_init_engines
