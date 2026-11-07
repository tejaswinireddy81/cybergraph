"""
modules/online_graph_stream.py
PHASE 3: Online Streaming Bipartite Graph Maintenance & GNN State Tracker

Maintains a sliding-window temporal bipartite graph (V_host, V_service, E) in memory.
Continuously updates unsupervised node features and updates GRU temporal state vectors
(h_{t-1} -> h_t) in real-time as new traffic flows arrive.
Also fixes GRU state persistence across topology changes and deterministic GNN edge
probability mapping (replaces unsafe probs[-1] assumption).
"""

import torch
import numpy as np
from modules.bipartite_graph_builder import BipartiteGraphBuilder


class OnlineGraphStream:
    """Sliding-Window Online Bipartite Graph & GNN State Tracker."""

    def __init__(self, gnn_model, scalers, window_size=500):
        self.gnn_model = gnn_model
        self.host_scaler = scalers["host_scaler"]
        self.service_scaler = scalers["service_scaler"]
        self.edge_scaler = scalers["edge_scaler"]
        self.window_size = window_size

        self.builder = BipartiteGraphBuilder()
        self.flow_buffer = []

        # ── GRU state persistence dictionaries ──
        # Maps stable node identity → hidden state tensor (last GRU output)
        # Key: node ID string (e.g. "HOST:192.168.1.1", "SERVICE:80/6")
        # Value: torch tensor of shape (hidden_dim,)
        # If a node is not in the dict, its state is initialized to zeros.
        self.host_states: dict = {}        # {host_node_id: torch_tensor}
        self.service_states: dict = {}     # {service_node_id: torch_tensor}
        self.edge_observation_counts = {}
    def ingest_flow(self, flow_rec):
        """Ingests a single flow record into the real-time graph stream."""
        self.flow_buffer.append(flow_rec)
        if len(self.flow_buffer) > self.window_size:
            self.flow_buffer.pop(0)

        self.builder.add_flow(
            src_ip=flow_rec["src_ip"],
            dst_ip=flow_rec["dst_ip"],
            dst_port=flow_rec["dst_port"],
            protocol=flow_rec["protocol"],
            timestamp=flow_rec["timestamp"],
            byte_count=flow_rec["byte_count"],
            pkt_count=flow_rec["pkt_count"],
            duration=flow_rec["duration"],
            xgb_prob=flow_rec["xgb_prob"],
            ae_score=flow_rec["ae_score"],
            fusion_score=flow_rec["fusion_score"],
            attack_label=flow_rec.get("attack_label", 0),
        )

        # ── Prune graph to bounded active window ──
        self._prune_to_active_window()

    def _prune_to_active_window(self):
        """Remove stale edges and isolated nodes outside the active flow window.

        Behaviour:
        - retain only nodes/edges represented by flows in the active window
        - remove stale edges (no flow in window)
        - remove isolated nodes
        - does NOT change the configured window_size
        """
        if not self.flow_buffer:
            return

        # Determine the set of active (host_node, service_node) pairs
        # represented by at least one flow in the window.
        active_edges = set()
        active_hosts = set()
        active_services = set()

        for flow in self.flow_buffer:
            src_ip = flow["src_ip"]
            dst_ip = flow["dst_ip"]
            dst_port = flow["dst_port"]
            protocol = flow["protocol"]

            host_node = self.builder.make_host_id(src_ip)
            service_node = self.builder.make_service_id(dst_port, protocol)
            edge_key = (host_node, service_node)

            self.edge_observation_counts[edge_key] = (self.edge_observation_counts.get(edge_key, 0) + 1)
            active_edges.add((host_node, service_node))
            active_edges.add((service_node, host_node))  # undirected check
            active_hosts.add(host_node)
            active_services.add(service_node)

        # Remove edges not in the active set
        for u, v in list(self.builder.g.edges()):
            if (u, v) not in active_edges and (v, u) not in active_edges:
                self.builder.g.remove_edge(u, v)

        # Remove isolated nodes
        for node in list(self.builder.g.nodes()):
            degree = self.builder.g.degree(node)
            if degree == 0:
                self.builder.g.remove_node(node)
                # Clean stats for removed node
                if node in self.builder._stats:
                    del self.builder._stats[node]

        # Clean stale GRU states for removed nodes
        for node_id in list(self.host_states.keys()):
            if node_id not in active_hosts:
                del self.host_states[node_id]
        for node_id in list(self.service_states.keys()):
            if node_id not in active_services:
                del self.service_states[node_id]

    # ── GRU state persistence accessors ──────────────────────────────────

    def _build_prev_host_h_tensor(self, h_map: dict, hidden_dim: int, device) -> torch.Tensor:
        """Build the prev_host_h tensor indexed by position matching the current graph tensor.

        h_map: {host_node_id: tensor_index_position} as produced by to_pyg_hetero()
        Returns tensor of shape (num_hosts, hidden_dim) where entry i is the GRU state
        for the host at position i (zero if host is new/not in state dict).
        """
        num_hosts = len(h_map)
        prev_h = torch.zeros(num_hosts, hidden_dim, device=device)

        # Sort host nodes to match the deterministic order produced by to_pyg_hetero()
        sorted_host_ids = sorted(h_map.keys())
        # h_map maps host_id → index in tensor
        for host_id in sorted_host_ids:
            if host_id in self.host_states:
                idx = h_map[host_id]
                prev_h[idx] = self.host_states[host_id]

        return prev_h

    def _build_prev_service_h_tensor(self, s_map: dict, hidden_dim: int, device) -> torch.Tensor:
        """Build the prev_service_h tensor indexed by position matching the current graph tensor.

        s_map: {service_node_id: tensor_index_position} as produced by to_pyg_hetero()
        Returns tensor of shape (num_services, hidden_dim) where entry j is the GRU state
        for the service at position j (zero if service is new/not in state dict).
        """
        num_services = len(s_map)
        prev_s = torch.zeros(num_services, hidden_dim, device=device)

        # Sort service nodes to match the deterministic order produced by to_pyg_hetero()
        sorted_service_ids = sorted(s_map.keys())
        for service_id in sorted_service_ids:
            if service_id in self.service_states:
                idx = s_map[service_id]
                prev_s[idx] = self.service_states[service_id]

        return prev_s

    # ── GNN inference ────────────────────────────────────────────────────

    def evaluate_realtime_gnn_risk(self):
        """
        Executes real-time Dynamic Bipartite Temporal GNN inference on the current online graph.

        Returns:
            probs:       numpy array of sigmoid edge probabilities (1 per edge)
            pyg_data:    PyG HeteroData object
            h_map:       dict mapping host_node_id → tensor index
            s_map:       dict mapping service_node_id → tensor index
            edge_prob_map: dict {(host_node_id, service_node_id): probability}
                             — replaces unsafe probs[-1] indexing
        """
        pyg_data, h_map, s_map = self.builder.to_pyg_hetero()

        if ("host", "connects_to", "service") not in pyg_data.edge_types or pyg_data["host", "connects_to", "service"].edge_index.shape[1] == 0:
            return np.empty((0,), dtype=np.float32), pyg_data, h_map, s_map, {}

        hx_numpy = pyg_data["host"].x.numpy()
        sx_numpy = pyg_data["service"].x.numpy()
        eidx_numpy = pyg_data["host", "connects_to", "service"].edge_index.numpy()
        eattr_numpy = pyg_data["host", "connects_to", "service"].edge_attr.numpy()

        hx_scaled = torch.tensor(self.host_scaler.transform(hx_numpy), dtype=torch.float32)
        sx_scaled = torch.tensor(self.service_scaler.transform(sx_numpy), dtype=torch.float32)
        eidx = torch.tensor(eidx_numpy, dtype=torch.long)
        eattr_scaled = torch.tensor(
    self.edge_scaler.transform(eattr_numpy),
    dtype=torch.float32)
        #print("[GNN EDGE FEATURE DEBUG]",    "raw min=", float(eattr_numpy.min()),"raw max=", float(eattr_numpy.max()),"scaled min=", float(eattr_scaled.min()),"scaled max=", float(eattr_scaled.max()),)

        #print("[GNN EDGE FEATURE ROW 0 RAW]",eattr_numpy[0].tolist())

        #print("[GNN EDGE FEATURE ROW 0 SCALED]",eattr_scaled[0].tolist())

        hidden_dim = self.gnn_model.temporal_update.host_gru.hidden_size

        # Build prev GRU states from persistence dictionaries
        prev_host_h = self._build_prev_host_h_tensor(h_map, hidden_dim, hx_scaled.device)
        prev_service_h = self._build_prev_service_h_tensor(s_map, hidden_dim, sx_scaled.device)

        self.gnn_model.eval()
        with torch.no_grad():
            logits, next_h_h, next_h_s = self.gnn_model(
                hx_scaled, sx_scaled, eidx, eattr_scaled, prev_host_h, prev_service_h
            )
            probs = torch.sigmoid(logits).numpy()

        # ── Store updated GRU states for persistence ──────────────────────
        # Store per-node states keyed by stable host/service identity.
        # The GNN returns next_h_h of shape (num_hosts, hidden_dim) and next_h_s
        # of shape (num_services, hidden_dim). We clone each row into the dict.

        # Reset state dict entries for nodes that now exist; keep old entries for
        # nodes that have disappeared (they will be pruned later).
        sorted_host_ids = sorted(h_map.keys())
        sorted_service_ids = sorted(s_map.keys())

        for host_id in sorted_host_ids:
            idx = h_map[host_id]
            self.host_states[host_id] = next_h_h[idx].clone()
        for service_id in sorted_service_ids:
            idx = s_map[service_id]
            self.service_states[service_id] = next_h_s[idx].clone()

        # ── Build edge key → probability mapping (replaces probs[-1]) ─────
        # Build O(1) inverted index: tensor_index → node_id
        inv_h_map = {idx: nid for nid, idx in h_map.items()}
        inv_s_map = {idx: sid for sid, idx in s_map.items()}

        edge_prob_map = {}
        edge_index = eidx  # shape (2, num_edges)
        edge_probs = probs.flatten()  # shape (num_edges,)

        for edge_idx in range(edge_index.shape[1]):
            h_idx = int(edge_index[0, edge_idx])
            s_idx = int(edge_index[1, edge_idx])

            host_id = inv_h_map.get(h_idx)
            service_id = inv_s_map.get(s_idx)

            if host_id is not None and service_id is not None:
                raw_gnn_risk = float(edge_probs[edge_idx])
                edge_key = (host_id, service_id)
                observations = self.edge_observation_counts.get(edge_key, 0) + 1
                self.edge_observation_counts[edge_key] = observations


# A first-seen edge has insufficient temporal evidence.
# Preserve the raw GNN score separately, but do not allow
# an isolated first observation to become a high-confidence
# graph threat.
                if observations <= 1:
                    effective_gnn_risk = 0.0
                else:
                    effective_gnn_risk = raw_gnn_risk
                edge_prob_map[edge_key] = effective_gnn_risk

        return probs, pyg_data, h_map, s_map, edge_prob_map

    def get_graph_summary(self):
        return self.builder.get_graph_summary()