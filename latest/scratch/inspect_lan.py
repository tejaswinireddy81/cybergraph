import ipaddress
import subprocess
import psutil
from scapy.all import conf

print("=== SCAPY INTERFACE ===")
print("conf.iface:", conf.iface)
print("conf.iface name:", getattr(conf.iface, "name", None))
print("conf.iface ip:", getattr(conf.iface, "ip", None))
print("conf.iface mac:", getattr(conf.iface, "mac", None))

print("\n=== SCAPY ROUTE ===")
try:
    print("conf.route.route('0.0.0.0'):", conf.route.route("0.0.0.0"))
except Exception as e:
    print("conf.route error:", e)

print("\n=== PSUTIL INTERFACES ===")
for iface_name, addrs in psutil.net_if_addrs().items():
    ipv4s = [a for a in addrs if getattr(a, "family", None) == 2]
    if ipv4s:
        print(f"Interface '{iface_name}':")
        for a in ipv4s:
            print(f"  address: {a.address}, netmask: {a.netmask}, broadcast: {a.broadcast}")

print("\n=== SYSTEM ARP -A ===")
try:
    arp_out = subprocess.check_output("arp -a", shell=True, text=True, stderr=subprocess.DEVNULL)
    print(arp_out[:1000])
except Exception as e:
    print("arp error:", e)
