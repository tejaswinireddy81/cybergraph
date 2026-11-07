import ipaddress
import psutil
import subprocess
from scapy.all import conf

def get_active_network_profile():
    local_ip = None
    default_gateway = None
    iface_name = None
    netmask = None

    # Method 1: Scapy routing table for 0.0.0.0 (primary default route)
    try:
        def_route = conf.route.route("0.0.0.0")
        if def_route and len(def_route) >= 3:
            # def_route is (iface, local_ip, gateway)
            cand_iface = def_route[0]
            cand_local = str(def_route[1])
            cand_gw = str(def_route[2])

            if cand_local and cand_local not in ("0.0.0.0", "127.0.0.1"):
                local_ip = cand_local
            if cand_gw and cand_gw not in ("0.0.0.0", "127.0.0.1", "255.255.255.255"):
                default_gateway = cand_gw
            if cand_iface:
                iface_name = getattr(cand_iface, "name", str(cand_iface))
    except Exception as e:
        print("Scapy route lookup error:", e)

    # Method 2: System routing table fallback if gateway/local_ip missing
    if not local_ip or not default_gateway:
        try:
            out = subprocess.check_output("netstat -rn", shell=True, text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0] == "0.0.0.0" and parts[1] == "0.0.0.0":
                    cand_gw = parts[2]
                    cand_local = parts[3]
                    if not default_gateway and cand_gw not in ("0.0.0.0", "127.0.0.1"):
                        default_gateway = cand_gw
                    if not local_ip and cand_local not in ("0.0.0.0", "127.0.0.1"):
                        local_ip = cand_local
                    break
        except Exception as e:
            print("netstat lookup error:", e)

    # Find netmask from psutil for the matched local_ip
    local_subnet = None
    try:
        for if_name, addrs in psutil.net_if_addrs().items():
            for snic in addrs:
                if getattr(snic, "family", None) == 2:
                    if snic.address == local_ip:
                        netmask = snic.netmask or "255.255.255.0"
                        if not iface_name or "{" in iface_name:
                            iface_name = if_name
                        break
            if netmask:
                break
    except Exception as e:
        print("psutil lookup error:", e)

    if local_ip:
        mask_to_use = netmask or "255.255.255.0"
        try:
            local_subnet = ipaddress.IPv4Network(f"{local_ip}/{mask_to_use}", strict=False)
        except Exception:
            local_subnet = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)

    return {
        "iface": iface_name or getattr(conf.iface, "name", "Wi-Fi"),
        "local_ip": local_ip,
        "netmask": netmask,
        "local_subnet": local_subnet,
        "default_gateway": default_gateway,
    }

prof = get_active_network_profile()
print("Network Profile Result:")
for k, v in prof.items():
    print(f"  {k}: {v} ({type(v).__name__})")
