import subprocess
import ipaddress
import re

def test_arp_discovery():
    try:
        out = subprocess.check_output("arp -a", shell=True, text=True, stderr=subprocess.DEVNULL)
        print("Raw arp -a output:\n", out)

        # Parse arp -a output by interface
        current_iface_ip = None
        iface_entries = {}
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("Interface:"):
                # e.g. Interface: 192.168.0.4 --- 0x11
                parts = line.split()
                if len(parts) >= 2:
                    current_iface_ip = parts[1]
                    iface_entries[current_iface_ip] = []
                continue

            # Format: Internet Address      Physical Address      Type
            #         192.168.0.1           b8-fb-b3-e6-a1-72     dynamic
            parts = line.split()
            if len(parts) >= 3 and current_iface_ip:
                ip_str = parts[0]
                mac_str = parts[1]
                entry_type = parts[2].lower()
                iface_entries[current_iface_ip].append({
                    "ip": ip_str,
                    "mac": mac_str,
                    "type": entry_type
                })

        print("Parsed iface_entries:", iface_entries)
    except Exception as e:
        print("ARP parse error:", e)

test_arp_discovery()
