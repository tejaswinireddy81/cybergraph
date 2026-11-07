"""
modules/dynamic_temporal_gnn.py
===============================
PHASE 5: Dynamic Bipartite Temporal Graph Neural Network (GNN)

Learns dynamic relationships between network entities (Hosts and Services)
across multiple flows and multiple temporal graph snapshots (G_1 -> G_2 -> ... -> G_K).

Architecture Components:
  1. HeteroBipartiteConv: Bipartite graph convolution aggregating structural context
     between Host nodes and Service nodes.
  2. TemporalNodeUpdate: GRU-based temporal recurrent state updater preserving entity
     behavioral memory across time snapshot steps (h_{t-1} -> h_t).
  3. EdgeRiskClassifier: Deep edge classification head predicting evolving attack probabilities
     for flow communications based on source host state, target service state, and edge features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroBipartiteConv(nn.Module):
    """Bipartite Graph Convolution aggregating Host <-> Service messages."""

    def __init__(self, host_dim, service_dim, hidden_dim):
        super().__init__()
        self.host_to_service = nn.Linear(host_dim, hidden_dim)
        self.service_to_host = nn.Linear(service_dim, hidden_dim)
        self.host_self = nn.Linear(host_dim, hidden_dim)
        self.service_self = nn.Linear(service_dim, hidden_dim)

    def forward(self, host_x, service_x, edge_index):
        num_hosts = host_x.shape[0]
        num_services = service_x.shape[0]
        hidden_dim = self.host_self.out_features

        msg_to_service = torch.zeros((num_services, hidden_dim), device=host_x.device)
        msg_to_host = torch.zeros((num_hosts, hidden_dim), device=host_x.device)

        if edge_index.shape[1] > 0:
            h_idx, s_idx = edge_index[0], edge_index[1]

            h_trans = F.relu(self.host_to_service(host_x))
            s_trans = F.relu(self.service_to_host(service_x))

            msg_to_service.index_add_(0, s_idx, h_trans[h_idx])
            msg_to_host.index_add_(0, h_idx, s_trans[s_idx])

            s_deg = torch.zeros((num_services, 1), device=host_x.device).index_add_(
                0, s_idx, torch.ones((len(s_idx), 1), device=host_x.device)
            ).clamp(min=1.0)
            h_deg = torch.zeros((num_hosts, 1), device=host_x.device).index_add_(
                0, h_idx, torch.ones((len(h_idx), 1), device=host_x.device)
            ).clamp(min=1.0)

            msg_to_service = msg_to_service / s_deg
            msg_to_host = msg_to_host / h_deg

        out_host = F.relu(self.host_self(host_x) + msg_to_host)
        out_service = F.relu(self.service_self(service_x) + msg_to_service)

        return out_host, out_service


class TemporalNodeUpdate(nn.Module):
    """GRU Temporal Recurrent Unit updating node memory across snapshots."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.host_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.service_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, curr_host, curr_service, prev_host_h=None, prev_service_h=None):
        if prev_host_h is None or prev_host_h.shape != curr_host.shape:
            prev_host_h = torch.zeros_like(curr_host)
        if prev_service_h is None or prev_service_h.shape != curr_service.shape:
            prev_service_h = torch.zeros_like(curr_service)

        next_host_h = self.host_gru(curr_host, prev_host_h)
        next_service_h = self.service_gru(curr_service, prev_service_h)

        return next_host_h, next_service_h


class EdgeRiskClassifier(nn.Module):
    """Edge Classification MLP predicting evolving attack probability."""

    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        in_dim = hidden_dim * 2 + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, host_states, service_states, edge_index, edge_attr):
        if edge_index.shape[1] == 0:
            return torch.empty((0, 1), device=host_states.device)

        h_idx, s_idx = edge_index[0], edge_index[1]
        h_emb = host_states[h_idx]
        s_emb = service_states[s_idx]

        cat_feat = torch.cat([h_emb, s_emb, edge_attr], dim=1)
        logits = self.mlp(cat_feat).squeeze(-1)
        return logits


class DynamicBipartiteTemporalGNN(nn.Module):
    """Master Dynamic Bipartite Temporal GNN Model."""

    def __init__(self, host_dim=8, service_dim=8, edge_dim=9, hidden_dim=32):
        super().__init__()
        self.conv1 = HeteroBipartiteConv(host_dim, service_dim, hidden_dim)
        self.temporal_update = TemporalNodeUpdate(hidden_dim)
        self.classifier = EdgeRiskClassifier(hidden_dim, edge_dim)

    def forward(self, host_x, service_x, edge_index, edge_attr, prev_host_h=None, prev_service_h=None):
        h_conv, s_conv = self.conv1(host_x, service_x, edge_index)
        h_state, s_state = self.temporal_update(h_conv, s_conv, prev_host_h, prev_service_h)
        cat_debug = torch.cat(
    [
        h_state[edge_index[0]],
        s_state[edge_index[1]],
        edge_attr
    ],
    dim=1
)

        #print(
    #"[GNN CLASSIFIER INPUT DEBUG]",
    #"min=", float(cat_debug.min()),
    #"max=", float(cat_debug.max()),
    #"mean=", float(cat_debug.mean()),
    #"std=", float(cat_debug.std()))

       # print( "[GNN STATE DEBUG]","host min/max=", float(h_state.min()), float(h_state.max()), "service min/max=", float(s_state.min()), float(s_state.max()))

        logits = self.classifier(h_state, s_state, edge_index, edge_attr)
        return logits, h_state, s_state
