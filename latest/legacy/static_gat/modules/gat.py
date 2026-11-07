import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=64, output_dim=2):

        super().__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=4)

        self.gat2 = GATConv(
            hidden_dim * 4,
            output_dim,
            heads=1,
            concat=False
        )

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)

        return x