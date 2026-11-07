"""
Maintains a live network topology.

Nodes = Devices
Edges = Communications

Each node stores:
- Risk
- Status
- Threat
- Last Seen

Used by the SOC dashboard to visualize
live network communication.
"""

import networkx as nx
from datetime import datetime


class TopologyEngine:

    def __init__(self):
        self.graph = nx.Graph()

    # ----------------------------------------------------
    # Main update method (used by inference.py)
    # ----------------------------------------------------
    def update(
        self,
        src_ip,
        dst_ip,
        risk=0,
        status="Normal",
        threat="None"
    ):
        self.update_connection(
            src_ip=src_ip,
            dst_ip=dst_ip,
            risk=risk,
            status=status,
            threat=threat
        )

    # ----------------------------------------------------
    # Backward-compatible method
    # ----------------------------------------------------
    def update_connection(
        self,
        src_ip,
        dst_ip,
        risk=0,
        status="Normal",
        threat="None"
    ):

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for ip in [src_ip, dst_ip]:

            if ip not in self.graph:

                self.graph.add_node(
                    ip,
                    risk=risk,
                    status=status,
                    threat=threat,
                    first_seen=now,
                    last_seen=now
                )

            else:

                self.graph.nodes[ip]["risk"] = risk
                self.graph.nodes[ip]["status"] = status
                self.graph.nodes[ip]["threat"] = threat
                self.graph.nodes[ip]["last_seen"] = now

        if self.graph.has_edge(src_ip, dst_ip):

            self.graph[src_ip][dst_ip]["count"] += 1

        else:

            self.graph.add_edge(
                src_ip,
                dst_ip,
                count=1
            )

    def get_graph(self):
        return self.graph

    def get_nodes(self):

        nodes = []

        for node in self.graph.nodes:

            data = self.graph.nodes[node]

            nodes.append({
                "ip": node,
                "risk": data.get("risk", 0),
                "status": data.get("status", "Normal"),
                "threat": data.get("threat", "None"),
                "last_seen": data.get("last_seen", "")
            })

        return nodes

    def get_edges(self):

        edges = []

        for src, dst, data in self.graph.edges(data=True):

            edges.append({
                "source": src,
                "target": dst,
                "count": data.get("count", 1)
            })

        return edges

    def total_devices(self):
        return self.graph.number_of_nodes()

    def total_connections(self):
        return self.graph.number_of_edges()

    def high_risk_devices(self):

        devices = []

        for node in self.graph.nodes:

            if self.graph.nodes[node]["risk"] >= 0.7:
                devices.append(node)

        return devices

    def clear(self):
        self.graph.clear()