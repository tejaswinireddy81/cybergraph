"""
firewall.py
===========
IP-level containment for the response engine. Two distinct mechanisms,
matching the severity table in the requirements doc:

  - Blacklist (soft): an in-process/JSON-persisted list with an
    expiry. Does NOT touch the OS firewall. Used for High severity
    ("temporarily blacklist the source IP") — it changes how the rest
    of the system treats the host (dashboard flags it, response engine
    won't re-alert on cooldown, etc.) without taking network-level
    action that could be wrong and hard to undo.

  - Block (hard): an actual Windows Firewall rule via `netsh advfirewall`,
    used for Critical severity after confidence verification. This is
    real enforcement, not a simulation — but because that's a real,
    potentially disruptive system change, it is gated behind two
    independent switches that both default to safe:

      1. `enabled`  — master kill switch, default False.
      2. `dry_run`  — log the exact command that *would* run, default True.

    Both must be explicitly turned off for this to touch the live
    firewall. On non-Windows hosts, blocking always runs in simulated
    mode regardless of these flags — there's no cross-platform
    enforcement backend implemented here, and pretending otherwise
    would be dishonest. (A Linux path via `iptables`/`nftables` or a
    cloud security-group API is a reasonable follow-up, but it has its
    own privilege and idempotency concerns and isn't included here
    until it's been built and tested against a real host.)

Every action, real or simulated, is logged to logs/firewall_actions.jsonl
so the incident/report layer has a full audit trail either way.
"""

import os
import ipaddress
import json
import platform
import subprocess
import threading
import time
import logging

logger = logging.getLogger(__name__)

_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
_ACTIONS_LOG = os.path.join(_LOG_DIR, "firewall_actions.jsonl")
_BLACKLIST_STORE = os.path.join(_LOG_DIR, "blacklist.json")

RULE_PREFIX = "NIDS_BLOCK_"
DEFAULT_BLACKLIST_TTL_SECONDS = 3600  # 1 hour "temporary" blacklist per the requirements doc


def _is_valid_ip(ip: str) -> bool:
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


class FirewallController:
    # Hardcoded whitelist to prevent self-inflicted DoS / locking out local host
    WHITELIST = {"127.0.0.1", "::1", "0.0.0.0", "localhost"}

    def __init__(
        self,
        enabled: bool = False,
        dry_run: bool = True,
        mode: str = None,
        blacklist_ttl_seconds: int = DEFAULT_BLACKLIST_TTL_SECONDS,
        max_blocks_per_minute: int = 10,
    ):
        os.makedirs(_LOG_DIR, exist_ok=True)
        
        # Explicit mode toggle: "dry-run" or "enforce"
        if mode is not None:
            self.mode = mode.lower()
            self.enabled = (self.mode == "enforce")
            self.dry_run = (self.mode == "dry-run")
        else:
            self.enabled = enabled
            self.dry_run = dry_run
            self.mode = "enforce" if (enabled and not dry_run) else "dry-run"

        self.blacklist_ttl_seconds = blacklist_ttl_seconds
        self.max_blocks_per_minute = max_blocks_per_minute
        self.system = platform.system()
        self.is_windows = (self.system == "Windows")
        self.is_linux = (self.system == "Linux")

        self._lock = threading.RLock()
        self._blocked = set()                 # IPs with a real/simulated firewall rule
        self._blacklist = {}                  # ip -> expiry_epoch (soft, non-OS-level)
        self._block_history = []              # timestamps of recent block actions for rate-limiting
        self._load_blacklist()

    def set_mode(self, mode: str):
        mode = mode.lower()
        if mode not in ["dry-run", "enforce"]:
            raise ValueError("mode must be 'dry-run' or 'enforce'")
        with self._lock:
            self.mode = mode
            self.enabled = (mode == "enforce")
            self.dry_run = (mode == "dry-run")
        logger.info("FirewallController mode updated to: %s", self.mode)

    # ── persistence for the soft blacklist (survives restarts) ────────

    def _load_blacklist(self):
        if os.path.exists(_BLACKLIST_STORE):
            try:
                with open(_BLACKLIST_STORE) as f:
                    self._blacklist = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._blacklist = {}

    def _save_blacklist(self):
        with open(_BLACKLIST_STORE, "w") as f:
            json.dump(self._blacklist, f, indent=2)

    def _log_action(self, record: dict):
        record["timestamp"] = time.time()
        with open(_ACTIONS_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _check_rate_limit(self) -> bool:
        """Returns True if block action is allowed; False if rate limit is exceeded."""
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            self._block_history = [t for t in self._block_history if t > cutoff]
            if len(self._block_history) >= self.max_blocks_per_minute:
                return False
            self._block_history.append(now)
            return True

    def is_whitelisted(self, ip: str) -> bool:
        if ip in self.WHITELIST or ip.startswith("127."):
            return True
        return False

    # ── soft blacklist (High severity) ─────────────────────────────────

    def blacklist_ip(self, ip: str, reason: str = "", ttl_seconds: int = None) -> dict:
        if not _is_valid_ip(ip):
            raise ValueError(f"Not a valid IP address: {ip!r}")
        if self.is_whitelisted(ip):
            return {"action": "blacklist", "ip": ip, "suppressed": True, "reason": "IP is whitelisted"}

        ttl = ttl_seconds or self.blacklist_ttl_seconds
        expiry = time.time() + ttl
        with self._lock:
            self._blacklist[ip] = expiry
            self._save_blacklist()
        record = {"action": "blacklist", "ip": ip, "reason": reason,
                  "ttl_seconds": ttl, "expires_at": expiry, "mode": self.mode}
        self._log_action(record)
        return record

    def is_blacklisted(self, ip: str) -> bool:
        with self._lock:
            expiry = self._blacklist.get(ip)
            if expiry is None:
                return False
            if expiry < time.time():
                del self._blacklist[ip]
                self._save_blacklist()
                return False
            return True

    def remove_from_blacklist(self, ip: str):
        with self._lock:
            if ip in self._blacklist:
                del self._blacklist[ip]
                self._save_blacklist()

    def active_blacklist(self) -> list:
        with self._lock:
            now = time.time()
            return [ip for ip, exp in self._blacklist.items() if exp >= now]

    # ── hard block (Critical severity) ─────────────────────────────────

    def block_ip(self, ip: str, reason: str = "") -> dict:
        """Attempts to install a real OS Firewall block rule (Windows netsh / Linux iptables/nftables)
        when mode is 'enforce'. If mode is 'dry-run', logs intended command without executing."""
        if not _is_valid_ip(ip):
            raise ValueError(f"Not a valid IP address: {ip!r}")

        if self.is_whitelisted(ip):
            record = {"action": "block", "ip": ip, "suppressed": True, "reason": "IP is whitelisted (loopback/gateway protection)"}
            self._log_action(record)
            return record

        if not self._check_rate_limit():
            record = {"action": "block", "ip": ip, "suppressed": True, "reason": "Rate-limit exceeded (max blocks/min reached)", "mode": self.mode}
            self._log_action(record)
            logger.warning("Firewall block suppressed for %s: rate limit exceeded.", ip)
            return record

        rule_name = f"{RULE_PREFIX}{ip.replace('.', '_').replace(':', '_')}"

        if self.mode == "dry-run" or self.dry_run or not self.enabled:
            return self._simulate_block(ip, rule_name, reason,
                                         note=f"dry-run mode enabled (showing command for OS: {self.system}).")

        # ── REAL ENFORCEMENT PATH (mode == 'enforce') ──
        commands = []
        if self.is_windows:
            commands = [
                ["netsh", "advfirewall", "firewall", "add", "rule",
                 f"name={rule_name}_IN", "dir=in", "action=block", f"remoteip={ip}"],
                ["netsh", "advfirewall", "firewall", "add", "rule",
                 f"name={rule_name}_OUT", "dir=out", "action=block", f"remoteip={ip}"],
            ]
        elif self.is_linux:
            # Try nftables first, fallback to iptables
            commands = [
                ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
                ["iptables", "-A", "OUTPUT", "-d", ip, "-j", "DROP"],
            ]

        results = []
        success = True
        for cmd in commands:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                results.append({"cmd": " ".join(cmd), "returncode": proc.returncode,
                                 "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})
                if proc.returncode != 0:
                    success = False
            except (subprocess.SubprocessError, OSError) as e:
                results.append({"cmd": " ".join(cmd), "error": str(e)})
                success = False

        with self._lock:
            if success:
                self._blocked.add(ip)

        record = {"action": "block", "ip": ip, "rule_name": rule_name, "reason": reason,
                  "mode": self.mode, "simulated": False, "success": success, "command_results": results}
        self._log_action(record)
        if not success:
            logger.error("Firewall block failed for %s: %s", ip, results)
        return record

    def _simulate_block(self, ip, rule_name, reason, note) -> dict:
        with self._lock:
            self._blocked.add(ip)
        record = {"action": "block", "ip": ip, "rule_name": rule_name, "reason": reason,
                  "mode": self.mode, "simulated": True, "success": True, "note": note}
        self._log_action(record)
        logger.info("Simulated firewall block for %s (%s)", ip, note)
        return record

    def unblock_ip(self, ip: str) -> dict:
        rule_name = f"{RULE_PREFIX}{ip.replace('.', '_').replace(':', '_')}"
        simulated = (self.mode == "dry-run" or self.dry_run or not self.enabled)

        if not simulated:
            commands = []
            if self.is_windows:
                commands = [
                    ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}_IN"],
                    ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}_OUT"],
                ]
            elif self.is_linux:
                commands = [
                    ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"],
                    ["iptables", "-D", "OUTPUT", "-d", ip, "-j", "DROP"],
                ]

            for cmd in commands:
                try:
                    subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                except (subprocess.SubprocessError, OSError) as e:
                    logger.error("Firewall unblock failed for %s: %s", ip, e)

        with self._lock:
            self._blocked.discard(ip)
        record = {"action": "unblock", "ip": ip, "rule_name": rule_name, "mode": self.mode, "simulated": simulated}
        self._log_action(record)
        return record

    def is_blocked(self, ip: str) -> bool:
        with self._lock:
            return ip in self._blocked

    def blocked_ips(self) -> list:
        with self._lock:
            return sorted(self._blocked)
