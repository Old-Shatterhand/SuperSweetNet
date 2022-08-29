from typing import Tuple, Union

import torch
import torch.nn.functional as F
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import glycan_to_graph
from pytorch_lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool as gmp, GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap


class SweetNetEncoder(LightningModule):
    """Wrapper for SweetNet that can be used with Lightning."""

    def __init__(
            self,
            graph_embed_dim,
            node_embed_dim,
            num_gnn_layers,
    ):
        super().__init__()

        # node embedding
        self.item_embedding = torch.nn.Embedding(num_embeddings=len(lib) + 1, embedding_dim=node_embed_dim)

        self.single_embed = torch.nn.Linear(node_embed_dim, graph_embed_dim)

        # convolution operations on the graph
        self.conv1 = GraphConv(node_embed_dim, node_embed_dim)
        self.pool1 = TopKPooling(node_embed_dim, ratio=1.0)
        self.conv2 = GraphConv(node_embed_dim, node_embed_dim)
        self.pool2 = TopKPooling(node_embed_dim, ratio=1.0)
        self.conv3 = GraphConv(node_embed_dim, node_embed_dim)
        self.pool3 = TopKPooling(node_embed_dim, ratio=1.0)

        self.lin = torch.nn.Linear(node_embed_dim * 2, graph_embed_dim)

    def forward(self, data):
        x, edge_index, batch = data["x"], data["edge_index"], data["batch"]

        # getting node features
        x = self.item_embedding(x)
        x = x.squeeze(1)

        # graph convolution operations
        x = F.leaky_relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.leaky_relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.leaky_relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # combining results from three graph convolutions
        x = x1 + x2 + x3

        x = self.lin(x)

        return x, None
