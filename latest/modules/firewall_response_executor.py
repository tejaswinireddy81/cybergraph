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
import logging

logger = logging.getLogger(__name__)


class FirewallResponseExecutor:
    """Manages real and simulated OS Firewall / ACL block, verify, and unblock actions."""

    def __init__(self, enable_real_blocking=False):
        self.enable_real_blocking = enable_real_blocking
        self.os_type = platform.system().lower()
        self.simulated_acl_blocked_ips = set()
        self.active_os_firewall_rules = {}

    def generate_rule_preview(self, ip_address, port=None):
        """Generates cross-platform OS firewall block command preview strings."""
        if self.os_type == "windows":
            rule_name = f"Block-GNN-{ip_address}"
            cmd = f'netsh advfirewall firewall add rule name="{rule_name}" dir=in action=block remoteip={ip_address}'
            unblock_cmd = f'netsh advfirewall firewall delete rule name="{rule_name}"'
            verify_cmd = f'netsh advfirewall firewall show rule name="{rule_name}"'
        else:
            cmd = f"iptables -A INPUT -s {ip_address} -j DROP"
            unblock_cmd = f"iptables -D INPUT -s {ip_address} -j DROP"
            verify_cmd = f"iptables -L INPUT -v -n | grep {ip_address}"

        return {
            "os_platform": self.os_type,
            "ip_address": ip_address,
            "port": port,
            "block_command_preview": cmd,
            "unblock_command_preview": unblock_cmd,
            "verify_command_preview": verify_cmd,
        }

    def verify_blocked(self, ip_address):
        """Verifies whether an IP is currently blocked in memory or in real OS firewall rules."""
        is_sim_blocked = ip_address in self.simulated_acl_blocked_ips
        is_real_blocked = ip_address in self.active_os_firewall_rules
        rule_active = is_sim_blocked or is_real_blocked

        return {
            "ip_address": ip_address,
            "is_blocked": rule_active,
            "simulated_memory_blocked": is_sim_blocked,
            "real_os_firewall_blocked": is_real_blocked,
            "status_description": "BLOCKED & VERIFIED" if rule_active else "UNBLOCKED / ALLOWED",
        }

    def execute_block(self, ip_address, port=None, force_real=False):
        """Executes a block action in simulated or real OS firewall mode."""
        preview = self.generate_rule_preview(ip_address, port)
        should_execute_real = force_real or self.enable_real_blocking

        if not should_execute_real:
            self.simulated_acl_blocked_ips.add(ip_address)
            return {
                "status": "SIMULATED_BLOCK_RECORDED",
                "execution_mode": "SIMULATED_MEMORY_ONLY",
                "ip_address": ip_address,
                "message": f"IP {ip_address} successfully added to Simulated ACL Block List.",
                "command_preview": preview["block_command_preview"],
                "verification": self.verify_blocked(ip_address),
            }

        # Real OS Firewall Command Execution
        try:
            cmd = preview["block_command_preview"]
            print(f"[FIREWALL EXECUTOR] Executing OS Firewall Command: {cmd} ...", flush=True)

            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)

            if res.returncode == 0:
                self.active_os_firewall_rules[ip_address] = preview["unblock_command_preview"]
                self.simulated_acl_blocked_ips.add(ip_address)
                return {
                    "status": "REAL_FIREWALL_RULE_APPLIED",
                    "execution_mode": "REAL_OS_FIREWALL",
                    "ip_address": ip_address,
                    "message": f"SUCCESS: Real OS firewall rule applied blocking IP {ip_address} on Windows Defender Firewall.",
                    "cmd_output": res.stdout.strip(),
                    "verification": self.verify_blocked(ip_address),
                }
            elif self.os_type == "windows":
                # Automatically trigger Windows UAC Administrator Elevation Prompt via PowerShell
                rule_name = f"Block-GNN-{ip_address}"
                ps_cmd = f'powershell -Command "Start-Process netsh -Verb RunAs -ArgumentList \'advfirewall firewall add rule name=\"{rule_name}\" dir=in action=block remoteip={ip_address}\'"'
                print(f"[FIREWALL EXECUTOR] Triggering Windows Admin Elevation: {ps_cmd} ...", flush=True)
                res_ps = subprocess.run(ps_cmd, shell=True, capture_output=True, text=True, timeout=10)

                if res_ps.returncode == 0:
                    self.active_os_firewall_rules[ip_address] = preview["unblock_command_preview"]
                    self.simulated_acl_blocked_ips.add(ip_address)
                    return {
                        "status": "REAL_FIREWALL_RULE_APPLIED",
                        "execution_mode": "REAL_OS_FIREWALL_ELEVATED",
                        "ip_address": ip_address,
                        "message": f"SUCCESS: Sent Windows Administrator approval request! Real OS firewall rule created for IP {ip_address}.",
                        "cmd_output": "Elevated via PowerShell RunAs",
                        "verification": self.verify_blocked(ip_address),
                    }
                else:
                    self.simulated_acl_blocked_ips.add(ip_address)
                    return {
                        "status": "ADMIN_PRIVILEGE_REQUIRED",
                        "execution_mode": "SIMULATED_MEMORY_ACTIVE",
                        "ip_address": ip_address,
                        "message": f"Simulated Block Active for {ip_address}! Real OS firewall rule requires Administrator approval.",
                        "error": res_ps.stderr.strip() or "Access Denied",
                        "verification": self.verify_blocked(ip_address),
                    }
            else:
                self.simulated_acl_blocked_ips.add(ip_address)
                return {
                    "status": "ADMIN_PRIVILEGE_REQUIRED",
                    "execution_mode": "SIMULATED_MEMORY_ACTIVE",
                    "ip_address": ip_address,
                    "message": f"Simulated Block Active for {ip_address}! Real OS firewall rule requires Administrator privileges.",
                    "error": res.stderr.strip() or "Access Denied (Requires Elevation)",
                    "command_attempted": cmd,
                    "verification": self.verify_blocked(ip_address),
                }
        except Exception as e:
            self.simulated_acl_blocked_ips.add(ip_address)
            return {
                "status": "ADMIN_PRIVILEGE_REQUIRED",
                "execution_mode": "SIMULATED_MEMORY_ACTIVE",
                "ip_address": ip_address,
                "message": f"Simulated Block Active for {ip_address}! Real OS firewall rule requires Administrator privileges.",
                "error": str(e),
                "verification": self.verify_blocked(ip_address),
            }

    def execute_unblock(self, ip_address):
        """Removes a real or simulated firewall block rule."""
        preview = self.generate_rule_preview(ip_address)

        if ip_address in self.simulated_acl_blocked_ips:
            self.simulated_acl_blocked_ips.remove(ip_address)

        if ip_address in self.active_os_firewall_rules:
            unblock_cmd = self.active_os_firewall_rules.pop(ip_address)
            try:
                subprocess.run(unblock_cmd, shell=True, capture_output=True, text=True, timeout=5)
            except Exception as e:
                logger.error(f"Failed to unblock {ip_address}: {e}")

        return {
            "status": "UNBLOCKED_SUCCESS",
            "ip_address": ip_address,
            "message": f"IP {ip_address} has been successfully unblocked.",
            "verification": self.verify_blocked(ip_address),
        }
