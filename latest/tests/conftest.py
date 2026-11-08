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
def _fake_os_firewall(monkeypatch):
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
    yield rules
