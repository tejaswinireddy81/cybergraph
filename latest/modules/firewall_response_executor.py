"""
modules/firewall_response_executor.py
======================================
OS FIREWALL & ACL RESPONSE EXECUTION MANAGER

Provides dual-mode firewall response capabilities:
  Mode 1: Simulated / Recommended Action (Default)
          Generates rule previews for Admin approval and maintains in-memory ACL block list.
  Mode 2: Real Active OS Firewall Execution (Optional Admin Action)
          Executes native OS firewall commands (`netsh advfirewall` on Windows / `iptables` on Linux)
          via subprocess with full administrative auditing and instant rule reversal (unblock).
"""

import os
import sys
import platform
import subprocess
import time
import logging
import ipaddress
import json
import threading

logger = logging.getLogger(__name__)


class FirewallResponseExecutor:
    """Manages real and simulated OS Firewall / ACL block, verify, and unblock actions.
    
    Hardened against command injection: validates all IP addresses via `ipaddress`
    and executes system processes with `shell=False` parameter lists.
    """

    MODE_DRY_RUN = "dry-run"
    MODE_ENFORCE = "enforce"
    VALID_MODES = (MODE_DRY_RUN, MODE_ENFORCE)
    DEFAULT_AUDIT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "firewall_audit.jsonl")

    def __init__(self, enable_real_blocking=None, mode=None, audit_log_path=None):
        """
        Safety model:
          * Default mode is "dry-run": NO OS firewall command is ever executed.
          * Real commands require mode == "enforce" (explicitly set via the admin API,
            CYBERGRAPH_FIREWALL_MODE=enforce, or mode="enforce") AND an admin-initiated
            call (allow_real=True). Automated alert responses never execute real commands.
        """
        if mode is None:
            if enable_real_blocking:
                mode = self.MODE_ENFORCE
            else:
                env_mode = os.environ.get("CYBERGRAPH_FIREWALL_MODE", "").strip().lower()
                mode = env_mode if env_mode in self.VALID_MODES else self.MODE_DRY_RUN
        mode = str(mode).strip().lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.audit_log_path = os.path.abspath(audit_log_path or self.DEFAULT_AUDIT_LOG)
        self._audit_lock = threading.Lock()
        self.os_type = platform.system().lower()
        self.simulated_acl_blocked_ips = set()
        self.active_os_firewall_rules = {}
        self.action_history = []
        self.protected_ips = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}  # nosec B104 - addresses that must NEVER be blocked, not a bind

    # ── mode handling ─────────────────────────────────────────────────

    @property
    def enable_real_blocking(self):
        """Backward-compatible read-only view: True only in explicit enforce mode."""
        return self.mode == self.MODE_ENFORCE

    @property
    def is_enforcing(self):
        return self.mode == self.MODE_ENFORCE

    def set_mode(self, mode, actor="system"):
        """Switch between 'dry-run' and 'enforce'. Raises ValueError on invalid mode or
        when leaving enforce mode while real OS rules are still active."""
        mode = str(mode).strip().lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        if mode == self.MODE_DRY_RUN and self.active_os_firewall_rules:
            raise ValueError(
                "Cannot switch to dry-run while real OS firewall rules are active "
                f"({', '.join(sorted(self.active_os_firewall_rules))}). Unblock them first."
            )
        previous = self.mode
        self.mode = mode
        self._audit({"action": "MODE_CHANGE", "ip": None, "previous_mode": previous,
                     "new_mode": mode, "status": "MODE_CHANGED"}, actor)
        return {"previous_mode": previous, "mode": mode}

    # ── persistent audit trail ────────────────────────────────────────

    def _audit(self, record, actor="system"):
        """Append one JSON line per block/unblock/mode-change. Never raises."""
        entry = {"timestamp": time.time(), "actor": actor or "system", "mode": self.mode}
        entry.update(record)
        try:
            os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
            with self._audit_lock:
                with open(self.audit_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + chr(10))
        except Exception as exc:
            logger.error("Could not write firewall audit log: %s", exc)
        logger.info("[FIREWALL AUDIT] %s", entry)

    def add_protected_ip(self, ip_str):
        """Register a critical IP (e.g. Gateway, DNS, Server IP) that must never be blocked."""
        if ip_str and self.validate_ip(ip_str):
            self.protected_ips.add(ip_str.strip())

    def is_protected(self, ip_str):
        """Check if an IP address belongs to the protected system/gateway/DNS whitelist."""
        clean = ip_str.strip() if ip_str else ""
        if clean in self.protected_ips:
            return True
        try:
            ip_obj = ipaddress.ip_address(clean)
            if ip_obj.is_loopback or ip_obj.is_multicast or ip_obj.is_reserved or ip_obj.is_link_local:
                return True
        except ValueError:
            pass
        return False

    def validate_ip(self, ip_address):
        """Validates that ip_address is a syntactically valid IPv4 or IPv6 address string."""
        if not ip_address or not isinstance(ip_address, str):
            return False
        try:
            ipaddress.ip_address(ip_address.strip())
            return True
        except (ValueError, AttributeError):
            return False

    def generate_rule_preview(self, ip_address, port=None):
        """Generates cross-platform OS firewall block command preview strings."""
        if not self.validate_ip(ip_address):
            return {
                "os_platform": self.os_type,
                "ip_address": ip_address,
                "port": port,
                "error": "Invalid IP address syntax",
                "block_command_preview": "INVALID_IP",
                "unblock_command_preview": "INVALID_IP",
                "verify_command_preview": "INVALID_IP",
            }

        clean_ip = ip_address.strip()
        if self.os_type == "windows":
            rule_name = f"Block-GNN-{clean_ip}"
            cmd = f'netsh advfirewall firewall add rule name="{rule_name}" dir=in action=block remoteip={clean_ip}'
            unblock_cmd = f'netsh advfirewall firewall delete rule name="{rule_name}"'
            verify_cmd = f'netsh advfirewall firewall show rule name="{rule_name}"'
        else:
            cmd = f"iptables -A INPUT -s {clean_ip} -j DROP"
            unblock_cmd = f"iptables -D INPUT -s {clean_ip} -j DROP"
            verify_cmd = f"iptables -L INPUT -v -n | grep {clean_ip}"

        return {
            "os_platform": self.os_type,
            "ip_address": clean_ip,
            "port": port,
            "block_command_preview": cmd,
            "unblock_command_preview": unblock_cmd,
            "verify_command_preview": verify_cmd,
        }

    def verify_blocked(self, ip_address):
        """Verifies whether an IP is currently blocked in memory or in real OS firewall rules."""
        if not self.validate_ip(ip_address):
            return {
                "ip_address": ip_address,
                "is_blocked": False,
                "simulated_memory_blocked": False,
                "real_os_firewall_blocked": False,
                "status_description": "INVALID_IP",
            }

        clean_ip = ip_address.strip()
        is_sim_blocked = clean_ip in self.simulated_acl_blocked_ips
        is_real_blocked = clean_ip in self.active_os_firewall_rules
        if self.os_type == "windows" and is_real_blocked:
            # Ground truth: ask the OS rather than trusting in-memory bookkeeping.
            is_real_blocked = self._windows_rule_exists(f"Block-GNN-{clean_ip}")
        rule_active = is_sim_blocked or is_real_blocked

        return {
            "ip_address": clean_ip,
            "is_blocked": rule_active,
            "simulated_memory_blocked": is_sim_blocked,
            "real_os_firewall_blocked": is_real_blocked,
            "status_description": "BLOCKED & VERIFIED" if rule_active else "UNBLOCKED / ALLOWED",
        }

    def execute_block(self, ip_address, port=None, force_real=False, allow_real=False, actor="system"):
        """Block an IP and write an audit record.

        A real OS command runs only if the call is admin-initiated (allow_real=True, or the
        legacy force_real=True) AND the executor is in explicit enforce mode. In dry-run a
        real request is downgraded to a simulated block and flagged in the result.
        """
        real_requested = bool(allow_real or force_real)
        res = self._execute_block_impl(ip_address, port=port, real_requested=real_requested)
        res.setdefault("mode", self.mode)
        if real_requested and not self.is_enforcing and res.get("execution_mode") == "SIMULATED_MEMORY_ONLY":
            res["dry_run"] = True
            res["real_block_requested"] = True
            res["message"] = (res.get("message", "") +
                              " [DRY-RUN: firewall is not in enforce mode, so no OS command was executed.]").strip()
        self._audit({"action": "BLOCK", "ip": str(ip_address).strip() if ip_address else ip_address,
                     "port": port, "real_requested": real_requested,
                     "status": res.get("status"), "execution_mode": res.get("execution_mode")}, actor)
        return res

    def _execute_block_impl(self, ip_address, port=None, real_requested=False):
        """Executes a block action in simulated or real OS firewall mode.
        
        Uses shell=False and parameter lists to eliminate command injection vulnerabilities.
        """
        if not self.validate_ip(ip_address):
            return {
                "status": "ERROR",
                "execution_mode": "REJECTED_VALIDATION",
                "ip_address": ip_address,
                "message": f"Rejected block action: '{ip_address}' is not a valid IP address.",
            }

        clean_ip = ip_address.strip()
        if self.is_protected(clean_ip):
            return {
                "status": "ERROR",
                "execution_mode": "PROTECTED_WHITELIST",
                "ip_address": clean_ip,
                "message": f"Rejected block action: '{clean_ip}' is a protected system/gateway/DNS address.",
            }

        preview = self.generate_rule_preview(clean_ip, port)
        should_execute_real = bool(real_requested and self.is_enforcing)

        if not should_execute_real:
            self.simulated_acl_blocked_ips.add(clean_ip)
            res = {
                "status": "SIMULATED_BLOCK_RECORDED",
                "execution_mode": "SIMULATED_MEMORY_ONLY",
                "ip_address": clean_ip,
                "message": f"IP {clean_ip} successfully added to Simulated ACL Block List.",
                "command_preview": preview["block_command_preview"],
                "verification": self.verify_blocked(clean_ip),
            }
            self.action_history.append({
                "action": "BLOCK",
                "ip": clean_ip,
                "port": port,
                "execution_mode": res["execution_mode"],
                "status": res["status"],
                "timestamp": time.time(),
            })
            return res

        # Real OS Firewall Command Execution without shell=True
        try:
            print(f"[FIREWALL EXECUTOR] Executing OS Firewall Command: {preview['block_command_preview']} ...", flush=True)

            if self.os_type == "windows":
                rule_name = f"Block-GNN-{clean_ip}"
                cmd_args = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name={rule_name}", "dir=in", "action=block", f"remoteip={clean_ip}"
                ]
            else:
                cmd_args = ["iptables", "-A", "INPUT", "-s", clean_ip, "-j", "DROP"]

            res = subprocess.run(cmd_args, shell=False, capture_output=True, text=True, timeout=5)

            if res.returncode == 0:
                self.active_os_firewall_rules[clean_ip] = preview["unblock_command_preview"]
                self.simulated_acl_blocked_ips.add(clean_ip)
                ret = {
                    "status": "REAL_FIREWALL_RULE_APPLIED",
                    "execution_mode": "REAL_OS_FIREWALL",
                    "ip_address": clean_ip,
                    "message": f"SUCCESS: Real OS firewall rule applied blocking IP {clean_ip}.",
                    "cmd_output": res.stdout.strip(),
                    "verification": self.verify_blocked(clean_ip),
                }
            elif self.os_type == "windows":
                # Automatically trigger Windows UAC Administrator Elevation Prompt via PowerShell without shell=True
                rule_name = f"Block-GNN-{clean_ip}"
                ps_args = [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    f"Start-Process netsh -Wait -Verb RunAs -ArgumentList 'advfirewall firewall add rule name=\"{rule_name}\" dir=in action=block remoteip={clean_ip}'"
                ]
                print(f"[FIREWALL EXECUTOR] Triggering Windows Admin Elevation: {rule_name} ...", flush=True)
                res_ps = subprocess.run(ps_args, shell=False, capture_output=True, text=True, timeout=60)

                # Start-Process returning 0 only means the launcher ran; confirm the rule really exists.
                if res_ps.returncode == 0 and self._windows_rule_exists(rule_name):
                    self.active_os_firewall_rules[clean_ip] = preview["unblock_command_preview"]
                    self.simulated_acl_blocked_ips.add(clean_ip)
                    ret = {
                        "status": "REAL_FIREWALL_RULE_APPLIED",
                        "execution_mode": "REAL_OS_FIREWALL_ELEVATED",
                        "ip_address": clean_ip,
                        "message": f"SUCCESS: Sent Windows Administrator approval request! Real OS firewall rule created for IP {clean_ip}.",
                        "cmd_output": "Elevated via PowerShell RunAs",
                        "verification": self.verify_blocked(clean_ip),
                    }
                else:
                    self.simulated_acl_blocked_ips.add(clean_ip)
                    ret = {
                        "status": "ADMIN_PRIVILEGE_REQUIRED",
                        "execution_mode": "SIMULATED_MEMORY_ACTIVE",
                        "ip_address": clean_ip,
                        "message": f"Simulated Block Active for {clean_ip}! Real OS firewall rule requires Administrator approval.",
                        "error": res_ps.stderr.strip() or "Administrator approval was declined or the rule was not created",
                        "verification": self.verify_blocked(clean_ip),
                    }
            else:
                self.simulated_acl_blocked_ips.add(clean_ip)
                ret = {
                    "status": "ADMIN_PRIVILEGE_REQUIRED",
                    "execution_mode": "SIMULATED_MEMORY_ACTIVE",
                    "ip_address": clean_ip,
                    "message": f"Simulated Block Active for {clean_ip}! Real OS firewall rule requires Administrator privileges.",
                    "error": res.stderr.strip() or "Access Denied (Requires Elevation)",
                    "command_attempted": preview["block_command_preview"],
                    "verification": self.verify_blocked(clean_ip),
                }
        except Exception as e:
            self.simulated_acl_blocked_ips.add(clean_ip)
            ret = {
                "status": "ADMIN_PRIVILEGE_REQUIRED",
                "execution_mode": "SIMULATED_MEMORY_ACTIVE",
                "ip_address": clean_ip,
                "message": f"Simulated Block Active for {clean_ip}! Real OS firewall rule requires Administrator privileges.",
                "error": str(e),
                "verification": self.verify_blocked(clean_ip),
            }

        self.action_history.append({
            "action": "BLOCK",
            "ip": clean_ip,
            "port": port,
            "execution_mode": ret.get("execution_mode", "UNKNOWN"),
            "status": ret.get("status", "UNKNOWN"),
            "timestamp": time.time(),
        })
        return ret

    def _windows_rule_exists(self, rule_name):
        """Return True if a Windows Firewall rule with this exact name exists."""
        try:
            r = subprocess.run(
                ["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule_name}"],
                shell=False, capture_output=True, text=True, timeout=10,
            )
            return r.returncode == 0 and "No rules match" not in r.stdout
        except Exception:
            return False

    def execute_unblock(self, ip_address, actor="system"):
        """Unblock an IP and write an audit record."""
        res = self._execute_unblock_impl(ip_address)
        res.setdefault("mode", self.mode)
        self._audit({"action": "UNBLOCK", "ip": str(ip_address).strip() if ip_address else ip_address,
                     "status": res.get("status"), "message": res.get("message")}, actor)
        return res

    def _execute_unblock_impl(self, ip_address):
        """Removes a real or simulated firewall block rule using shell=False."""
        if not self.validate_ip(ip_address):
            return {
                "status": "ERROR",
                "ip_address": ip_address,
                "message": f"Rejected unblock action: '{ip_address}' is not a valid IP address.",
            }

        clean_ip = ip_address.strip()
        if clean_ip in self.simulated_acl_blocked_ips:
            self.simulated_acl_blocked_ips.discard(clean_ip)

        had_real_rule = clean_ip in self.active_os_firewall_rules
        self.active_os_firewall_rules.pop(clean_ip, None)

        os_failure = None
        # OS commands only for rules this executor really created, and only in enforce mode.
        if had_real_rule and self.is_enforcing:
            try:
                if self.os_type == "windows":
                    rule_name = f"Block-GNN-{clean_ip}"
                    if self._windows_rule_exists(rule_name):
                        cmd_args = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"]
                        res = subprocess.run(cmd_args, shell=False, capture_output=True, text=True, timeout=5)
                        if res.returncode != 0 and self._windows_rule_exists(rule_name):
                            # Not elevated: retry through a UAC prompt, then confirm it is really gone.
                            ps_args = [
                                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                                f"Start-Process netsh -Wait -Verb RunAs -ArgumentList 'advfirewall firewall delete rule name=\"{rule_name}\"'",
                            ]
                            subprocess.run(ps_args, shell=False, capture_output=True, text=True, timeout=60)
                        if self._windows_rule_exists(rule_name):
                            os_failure = "OS firewall rule still present (Administrator approval required to delete it)."
                else:
                    cmd_args = ["iptables", "-D", "INPUT", "-s", clean_ip, "-j", "DROP"]
                    res = subprocess.run(cmd_args, shell=False, capture_output=True, text=True, timeout=5)
                    if res.returncode != 0:
                        os_failure = res.stderr.strip() or "iptables delete failed (requires root)."
            except Exception as e:
                logger.error(f"Failed to unblock {clean_ip}: {e}")
                os_failure = str(e)

        if os_failure:
            # Keep tracking the rule so the operator can retry, and report the truth.
            self.active_os_firewall_rules[clean_ip] = f"retry unblock {clean_ip}"
            ret = {
                "status": "UNBLOCK_FAILED",
                "ip_address": clean_ip,
                "message": f"Could not remove OS firewall rule for {clean_ip}: {os_failure}",
                "verification": self.verify_blocked(clean_ip),
            }
            self.action_history.append({
                "action": "UNBLOCK", "ip": clean_ip, "status": ret["status"], "timestamp": time.time(),
            })
            return ret

        ret = {
            "status": "UNBLOCKED_SUCCESS",
            "ip_address": clean_ip,
            "message": f"IP {clean_ip} has been successfully unblocked.",
            "verification": self.verify_blocked(clean_ip),
        }
        self.action_history.append({
            "action": "UNBLOCK",
            "ip": clean_ip,
            "status": ret["status"],
            "timestamp": time.time(),
        })
        return ret
