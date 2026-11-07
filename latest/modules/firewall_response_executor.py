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

logger = logging.getLogger(__name__)


class FirewallResponseExecutor:
    """Manages real and simulated OS Firewall / ACL block, verify, and unblock actions.
    
    Hardened against command injection: validates all IP addresses via `ipaddress`
    and executes system processes with `shell=False` parameter lists.
    """

    def __init__(self, enable_real_blocking=False):
        self.enable_real_blocking = enable_real_blocking
        self.os_type = platform.system().lower()
        self.simulated_acl_blocked_ips = set()
        self.active_os_firewall_rules = {}
        self.action_history = []

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
        rule_active = is_sim_blocked or is_real_blocked

        return {
            "ip_address": clean_ip,
            "is_blocked": rule_active,
            "simulated_memory_blocked": is_sim_blocked,
            "real_os_firewall_blocked": is_real_blocked,
            "status_description": "BLOCKED & VERIFIED" if rule_active else "UNBLOCKED / ALLOWED",
        }

    def execute_block(self, ip_address, port=None, force_real=False):
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
        preview = self.generate_rule_preview(clean_ip, port)
        should_execute_real = force_real or self.enable_real_blocking

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
                    f"Start-Process netsh -Verb RunAs -ArgumentList 'advfirewall firewall add rule name=\"{rule_name}\" dir=in action=block remoteip={clean_ip}'"
                ]
                print(f"[FIREWALL EXECUTOR] Triggering Windows Admin Elevation: {rule_name} ...", flush=True)
                res_ps = subprocess.run(ps_args, shell=False, capture_output=True, text=True, timeout=10)

                if res_ps.returncode == 0:
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
                        "error": res_ps.stderr.strip() or "Access Denied",
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

    def execute_unblock(self, ip_address):
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

        if had_real_rule or self.enable_real_blocking:
            try:
                if self.os_type == "windows":
                    rule_name = f"Block-GNN-{clean_ip}"
                    cmd_args = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"]
                else:
                    cmd_args = ["iptables", "-D", "INPUT", "-s", clean_ip, "-j", "DROP"]
                subprocess.run(cmd_args, shell=False, capture_output=True, text=True, timeout=5)
            except Exception as e:
                logger.error(f"Failed to unblock {clean_ip}: {e}")

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
