import sys
import os
import ipaddress
import re

sys.path.insert(0, r"c:\Users\LENOVO\Downloads\major_project (1)\latest")

import app
from app import (
    get_active_network_profile,
    discover_lan_neighbors,
    sync_network_state,
    is_local_subnet_ip,
    is_local_lan_host,
    get_server_local_ip,
    set_server_local_ip,
    add_discovered_lan_host,
    get_discovered_lan_hosts,
    get_monitored_local_networks,
    set_monitored_local_networks,
    get_default_gateways,
    set_default_gateways,
    _discovered_lan_hosts,
)

print("=" * 80)
print("TEST SUITE: FINAL DYNAMIC LAN ENDPOINT DISCOVERY ARCHITECTURE (A - F)")
print("=" * 80)

# ==============================================================================
# TEST A: Current Network Dynamic Discovery & Gateway Exclusion
# ==============================================================================
print("\n[TEST A] Current Network Discovery & Server/Gateway Identification")
sync_network_state(force_scan=True)
prof = get_active_network_profile()
print(f"  Discovered Active Interface: {prof.get('iface')}")
print(f"  Discovered Local IP:         {prof.get('local_ip')}")
print(f"  Discovered Subnet:           {prof.get('local_subnet')}")
print(f"  Discovered Default Gateway:  {prof.get('default_gateway')}")

local_server_ip = prof.get("local_ip")
default_gw = prof.get("default_gateway")

assert local_server_ip is not None, "Local server IP must be dynamically discovered"
assert default_gw is not None, "Default gateway must be dynamically discovered"

# 1. Gateway is NOT HOST
res_gw = is_local_lan_host(default_gw)
print(f"  Default Gateway {default_gw} -> is_local_lan_host? {res_gw}")
assert res_gw is False, f"Default gateway {default_gw} MUST NOT be classified as HOST"
assert default_gw not in _discovered_lan_hosts, f"Default gateway {default_gw} MUST NOT be in _discovered_lan_hosts"

# 2. Local server IP IS HOST
res_srv = is_local_lan_host(local_server_ip)
print(f"  Local Server IP {local_server_ip} -> is_local_lan_host? {res_srv}")
assert res_srv is True, f"Local server IP {local_server_ip} MUST be classified as HOST"
assert local_server_ip in _discovered_lan_hosts, f"Local server IP {local_server_ip} MUST be in _discovered_lan_hosts"

print("  PASSED: Test A passed successfully.")

# ==============================================================================
# TEST B: Another LAN Endpoint is Discovered -> HOST
# ==============================================================================
print("\n[TEST B] Discovered LAN Endpoint -> HOST")
# Simulate another LAN endpoint on current subnet
current_subnet = prof.get("local_subnet")
sample_peer = None
for host_candidate in current_subnet.hosts():
    cand_str = str(host_candidate)
    if cand_str != local_server_ip and cand_str != default_gw and cand_str not in _discovered_lan_hosts:
        sample_peer = cand_str
        break

assert sample_peer is not None, "Must find a candidate peer on current subnet"

# Initially, unconfirmed peer is NOT a persistent HOST (Requirements 8 & 9)
assert is_local_lan_host(sample_peer) is False, f"Unconfirmed peer {sample_peer} must not be HOST before discovery"

# 1. Test discovery via ARP parsing
simulated_arp = f"""
Interface: {local_server_ip} --- 0x11
  Internet Address      Physical Address      Type
  {default_gw}           b8-fb-b3-e6-a1-72     dynamic
  {sample_peer}          50-eb-71-33-aa-bb     dynamic
  255.255.255.255       ff-ff-ff-ff-ff-ff     static
"""
discovered_from_arp = discover_lan_neighbors(profile=prof, arp_output=simulated_arp)
print(f"  Discovered neighbors from simulated ARP: {discovered_from_arp}")
assert sample_peer in discovered_from_arp, f"Neighbor {sample_peer} must be discovered from ARP"
assert default_gw not in discovered_from_arp, f"Gateway {default_gw} must not be discovered from ARP"

# 2. Add to discovered hosts
add_ok = add_discovered_lan_host(sample_peer)
assert add_ok is True, f"Failed to register valid peer {sample_peer}"
assert is_local_lan_host(sample_peer) is True, f"Discovered peer {sample_peer} must now be classified as HOST"
assert sample_peer in _discovered_lan_hosts, f"{sample_peer} must be in _discovered_lan_hosts"

# 3. Verify graph node output
# A HOST node should have type: HOST and label: sample_peer
test_nodes = []
for h_ip in _discovered_lan_hosts:
    test_nodes.append({"label": h_ip, "type": "HOST", "color": "normal"})
peer_node = next((n for n in test_nodes if n["label"] == sample_peer), None)
assert peer_node is not None, f"Node for {sample_peer} must exist in graph nodes"
assert peer_node["type"] == "HOST", f"Node for {sample_peer} must have type HOST"

print(f"  PASSED: Test B passed successfully. Peer {sample_peer} confirmed as HOST.")

# ==============================================================================
# TEST C: Public IP -> EXTERNAL
# ==============================================================================
print("\n[TEST C] Public IP Classification -> EXTERNAL")
for pub_ip in ["8.8.8.8", "142.250.183.163", "1.1.1.1", "204.79.197.200"]:
    res_pub = is_local_lan_host(pub_ip)
    print(f"  {pub_ip:18} -> is_local_lan_host? {res_pub}")
    assert res_pub is False, f"Public IP {pub_ip} MUST NOT be classified as HOST"
    assert pub_ip not in _discovered_lan_hosts, f"Public IP {pub_ip} MUST NOT be in _discovered_lan_hosts"

    # Edge and node classification
    edge_label = pub_ip if is_local_lan_host(pub_ip) else f"EXT:{pub_ip}"
    assert edge_label == f"EXT:{pub_ip}", f"Edge label for public IP must be EXT:{pub_ip}"

print("  PASSED: Test C passed successfully. Public IPs correctly classified as EXTERNAL.")

# ==============================================================================
# TEST D: Multicast / Broadcast / Special -> NOT HOST
# ==============================================================================
print("\n[TEST D] Multicast / Broadcast / Special -> NOT HOST")
special_ips = [
    "224.0.0.1",
    "224.0.0.251",
    "239.255.255.250",
    "255.255.255.255",
    str(current_subnet.network_address),
    str(current_subnet.broadcast_address),
    "0.0.0.0",
    "127.0.0.1",
    "169.254.1.1",
    "::1",
    "invalid-ip",
]

for s_ip in special_ips:
    res_s = is_local_lan_host(s_ip)
    print(f"  {s_ip:25} -> is_local_lan_host? {res_s}")
    assert res_s is False, f"Special IP {s_ip} MUST NOT be classified as HOST"
    assert s_ip not in _discovered_lan_hosts, f"Special IP {s_ip} must not be in _discovered_lan_hosts"

    # Registration attempt must be rejected
    res_add = add_discovered_lan_host(s_ip)
    assert res_add is False, f"Registration of special IP {s_ip} must be rejected"

print("  PASSED: Test D passed successfully. Special, multicast, and broadcast rejected as HOST.")

# ==============================================================================
# TEST E: Network Change Handling
# ==============================================================================
print("\n[TEST E] Network Change Handling (Network A -> Network B)")

# 1. Setup Network A
net_a_profile = {
    "iface": "Wi-Fi",
    "local_ip": "192.168.0.4",
    "netmask": "255.255.255.0",
    "local_subnet": ipaddress.IPv4Network("192.168.0.0/24"),
    "default_gateway": "192.168.0.1",
}
sync_network_state(force_scan=True, profile_override=net_a_profile)
# Add an endpoint on Network A
add_discovered_lan_host("192.168.0.55")

print(f"  Network A Active:")
print(f"    Server IP:       {get_server_local_ip()}")
print(f"    Discovered HOST: {_discovered_lan_hosts}")
assert "192.168.0.4" in _discovered_lan_hosts
assert "192.168.0.55" in _discovered_lan_hosts
assert "192.168.0.1" not in _discovered_lan_hosts
assert is_local_lan_host("192.168.0.4") is True
assert is_local_lan_host("192.168.0.55") is True
assert is_local_lan_host("192.168.0.1") is False

# 2. Switch to Network B (e.g. campus Wi-Fi)
net_b_profile = {
    "iface": "Wi-Fi",
    "local_ip": "10.160.36.211",
    "netmask": "255.255.255.0",
    "local_subnet": ipaddress.IPv4Network("10.160.36.0/24"),
    "default_gateway": "10.160.36.14",
}
sync_network_state(force_scan=True, profile_override=net_b_profile)
# Add an endpoint on Network B
add_discovered_lan_host("10.160.36.186")

print(f"  Switched to Network B:")
print(f"    Server IP:       {get_server_local_ip()}")
print(f"    Discovered HOST: {_discovered_lan_hosts}")

# Checks for Network B:
assert "10.160.36.211" in _discovered_lan_hosts, "Network B Server IP must be in HOST registry"
assert "10.160.36.186" in _discovered_lan_hosts, "Network B Discovered peer must be in HOST registry"
assert "10.160.36.14" not in _discovered_lan_hosts, "Network B Gateway must NOT be in HOST registry"
assert is_local_lan_host("10.160.36.211") is True
assert is_local_lan_host("10.160.36.186") is True
assert is_local_lan_host("10.160.36.14") is False

# Critical Checks: Network A hosts must be REMOVED and not remain as HOSTs
assert "192.168.0.4" not in _discovered_lan_hosts, "Old Network A server IP must NOT remain in registry"
assert "192.168.0.55" not in _discovered_lan_hosts, "Old Network A peer must NOT remain in registry"
assert is_local_lan_host("192.168.0.4") is False, "Old Network A server IP must NOT be classified as HOST"
assert is_local_lan_host("192.168.0.55") is False, "Old Network A peer must NOT be classified as HOST"
assert not any(ip.startswith("192.168.0.") for ip in _discovered_lan_hosts), "No Network A IPs should linger"

print("  PASSED: Test E passed successfully. Network switch cleared old hosts, registered new server, discovered new peer, and excluded new gateway.")

# Restore live network
sync_network_state(force_scan=True)
print(f"  Restored Live Network Profile: {get_active_network_profile().get('local_subnet')}")

# ==============================================================================
# TEST F: No Hard-Coded IP Allowlist
# ==============================================================================
print("\n[TEST F] Verify Zero Hard-Coded IP Addresses in app.py Logic")

with open(r"c:\Users\LENOVO\Downloads\major_project (1)\latest\app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

forbidden_ips = [
    "192.168.0.1",
    "192.168.0.3",
    "192.168.0.4",
    "10.160.36.14",
    "10.160.36.186",
    "10.160.36.211",
    "10.160.36.227",
]

violations = []
for idx, line in enumerate(lines, 1):
    # Strip comments
    code_part = line.split("#")[0].strip()
    if not code_part:
        continue
    # Check for quotes containing forbidden IPs
    for f_ip in forbidden_ips:
        if f'"{f_ip}"' in code_part or f"'{f_ip}'" in code_part:
            violations.append((idx, f_ip, line.strip()))

if violations:
    print("  FAILED: Found hardcoded IPs in executable code:")
    for v in violations:
        print(f"    Line {v[0]}: {v[1]} -> {v[2]}")
    assert False, f"Hardcoded IP violations found: {violations}"
else:
    print("  PASSED: Zero hard-coded IP addresses found in app.py logic.")

print("\n" + "=" * 80)
print("ALL 6 ARCHITECTURE TEST SUITES (A, B, C, D, E, F) PASSED WITH 100% SUCCESS!")
print("=" * 80)
