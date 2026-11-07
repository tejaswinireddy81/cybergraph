"""
modules/temporal_graph_storage.py
==================================
PHASE 4: PyTorch Geometric-Compatible Temporal Graph Storage & Serializer

Maintains and serializes temporal bipartite graph snapshots (G_1 -> G_2 -> ... -> G_K).
Provides structured, PyTorch Geometric-compatible data objects (HeteroData / Data)
that can be reloaded directly by the future Dynamic Bipartite Temporal GNN (Phase 5)
without needing to rebuild from the raw dataset.
"""

import os
import json
import torch
import numpy as np

try:
    from torch_geometric.data import HeteroData, Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class TemporalGraphSnapshot:
    """Encapsulates a single temporal bipartite snapshot G_t."""

    def __init__(
        self,
        window_id,
        timestamp_start,
        timestamp_end,
        host_nodes,
        service_nodes,
        host_x,
        service_x,
        edge_index_host_service,
        edge_attr,
        edge_y,
        metadata=None,
    ):
        self.window_id = int(window_id)
        self.timestamp_start = float(timestamp_start)
        self.timestamp_end = float(timestamp_end)
        self.host_nodes = list(host_nodes)
        self.service_nodes = list(service_nodes)
        self.host_x = np.array(host_x, dtype=np.float32)
        self.service_x = np.array(service_x, dtype=np.float32)
        self.edge_index_host_service = np.array(edge_index_host_service, dtype=np.int64)
        self.edge_attr = np.array(edge_attr, dtype=np.float32)
        self.edge_y = np.array(edge_y, dtype=np.int64)
        self.metadata = metadata or {}

    def to_pyg_hetero(self):
        """Converts snapshot to torch_geometric.data.HeteroData."""
        data = HeteroData()
        data.window_id = self.window_id
        data.timestamp_start = self.timestamp_start
        data.timestamp_end = self.timestamp_end

        # Host and Service node feature matrices
        data["host"].x = torch.tensor(self.host_x, dtype=torch.float32)
        data["service"].x = torch.tensor(self.service_x, dtype=torch.float32)
        data["host"].node_ids = self.host_nodes
        data["service"].node_ids = self.service_nodes

        # Bipartite edge tensors: HOST -> SERVICE
        if self.edge_index_host_service.size > 0:
            data["host", "connects_to", "service"].edge_index = torch.tensor(
                self.edge_index_host_service, dtype=torch.long
            )
            data["host", "connects_to", "service"].edge_attr = torch.tensor(
                self.edge_attr, dtype=torch.float32
            )
            data["host", "connects_to", "service"].y = torch.tensor(
                self.edge_y, dtype=torch.long
            )
        else:
            data["host", "connects_to", "service"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["host", "connects_to", "service"].edge_attr = torch.empty((0, self.edge_attr.shape[1] if self.edge_attr.ndim > 1 else 0), dtype=torch.float32)
            data["host", "connects_to", "service"].y = torch.empty((0,), dtype=torch.long)

        return data

    def to_dict(self):
        return {
            "window_id": self.window_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "host_nodes": self.host_nodes,
            "service_nodes": self.service_nodes,
            "host_x": self.host_x.tolist(),
            "service_x": self.service_x.tolist(),
            "edge_index_host_service": self.edge_index_host_service.tolist(),
            "edge_attr": self.edge_attr.tolist(),
            "edge_y": self.edge_y.tolist(),
            "metadata": self.metadata,
        }


class TemporalGraphStorage:
    def __init__(self, snapshots=None):
        self.snapshots = snapshots or []

    def add_snapshot(self, snapshot: TemporalGraphSnapshot):
        self.snapshots.append(snapshot)

    def save(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Store as torch dictionary containing tensors and metadata
        dataset_payload = {
            "num_snapshots": len(self.snapshots),
            "snapshots": [s.to_dict() for s in self.snapshots],
        }
        torch.save(dataset_payload, filepath)
        print(f"Saved {len(self.snapshots)} temporal graph snapshots to: {filepath}")

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Temporal graph dataset file not found: {filepath}")
        payload = torch.load(filepath)
        storage = cls()
        for s_dict in payload["snapshots"]:
            snap = TemporalGraphSnapshot(
                window_id=s_dict["window_id"],
                timestamp_start=s_dict["timestamp_start"],
                timestamp_end=s_dict["timestamp_end"],
                host_nodes=s_dict["host_nodes"],
                service_nodes=s_dict["service_nodes"],
                host_x=s_dict["host_x"],
                service_x=s_dict["service_x"],
                edge_index_host_service=s_dict["edge_index_host_service"],
                edge_attr=s_dict["edge_attr"],
                edge_y=s_dict["edge_y"],
                metadata=s_dict.get("metadata"),
            )
            storage.add_snapshot(snap)
        print(f"Loaded {len(storage.snapshots)} temporal graph snapshots from: {filepath}")
        return storage

    def get_pyg_snapshot(self, window_idx):
        if window_idx < 0 or window_idx >= len(self.snapshots):
            raise IndexError(f"Snapshot index {window_idx} out of range (0-{len(self.snapshots)-1})")
        return self.snapshots[window_idx].to_pyg_hetero()

    def get_previous_current_next(self, window_idx):
        """Loads (prev_g, curr_g, next_g) triple for Temporal GNN sliding window input."""
        curr_g = self.get_pyg_snapshot(window_idx)
        prev_g = self.get_pyg_snapshot(window_idx - 1) if window_idx > 0 else None
        next_g = self.get_pyg_snapshot(window_idx + 1) if window_idx < len(self.snapshots) - 1 else None
        return prev_g, curr_g, next_g
