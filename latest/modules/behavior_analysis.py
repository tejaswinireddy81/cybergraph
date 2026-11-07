from collections import defaultdict, Counter
import ipaddress
import time


def _is_local(ip):
    """RFC1918/loopback/link-local check -- a physical host on this LAN
    rather than an internet endpoint. Duplicated (small) from app.py's
    _is_local_ip rather than imported, so this module has no dependency
    on the dashboard."""
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except (ValueError, TypeError):
        return False


class BehaviorAnalyzer:

    def __init__(self):
        self.devices = defaultdict(self.create_profile)

    def create_profile(self):
        now = time.time()
        return {
            "packets": 0,
            "bytes": 0,
            "flows": 0,
            "destinations": set(),
            "ports": set(),
            "first_seen": now,
            "last_seen": now,
            "risk_score": 0,
            # outgoing (this device as source) / incoming (as destination,
            # local devices only -- see update()) traffic split, for
            # Device Deep-Dive's "incoming vs outgoing" requirement
            "out_packets": 0,
            "out_bytes": 0,
            "in_packets": 0,
            "in_bytes": 0,
            # protocol distribution for this device's outgoing flows
            "protocols": Counter(),
            # dst_ip -> bytes sent, for a "top peers" breakdown
            "peers": Counter(),
            "is_local": None,
        }

    def update(self, src_ip, dst_ip, dst_port, packet_size, pkt_count=1, protocol=None):
        device = self.devices[src_ip]
        if device["is_local"] is None:
            device["is_local"] = _is_local(src_ip)

        device["packets"] += pkt_count
        device["bytes"] += packet_size
        device["flows"] += 1
        device["out_packets"] += pkt_count
        device["out_bytes"] += packet_size

        device["destinations"].add(dst_ip)
        device["ports"].add(dst_port)
        device["peers"][dst_ip] += packet_size
        if protocol is not None:
            device["protocols"][protocol] += 1

        device["last_seen"] = time.time()

        # Mirror the incoming side onto the destination's own profile, but
        # only when the destination is local -- otherwise every external
        # site a device talks to would spin up its own tracked profile,
        # which is neither useful nor bounded.
        if _is_local(dst_ip):
            peer_device = self.devices[dst_ip]
            if peer_device["is_local"] is None:
                peer_device["is_local"] = True
            peer_device["in_packets"] += pkt_count
            peer_device["in_bytes"] += packet_size
            peer_device["last_seen"] = time.time()

    def get_profile(self, ip):
        return self.devices[ip]

    def top_peers(self, ip, n=5):
        return self.devices[ip]["peers"].most_common(n)

    def protocol_distribution(self, ip):
        return dict(self.devices[ip]["protocols"])
