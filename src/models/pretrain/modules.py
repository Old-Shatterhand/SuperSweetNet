from typing import Union, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.models.base_model import MLP


class PositionalEncoder(LightningModule):
    def __init__(
            self,
            graph_embed_dim: int,
            num_gnn_layers: int,
            **kwargs,
    ):
        super(PositionalEncoder, self).__init__()
        self.feat_embed = MLP(3, [64], graph_embed_dim)
        self.node_embed = torch.nn.ModuleList([
            GCNConv(graph_embed_dim, graph_embed_dim) for _ in range(num_gnn_layers)
        ])

    def forward(self, data: Union[dict, Data]) -> Tuple[Tensor, Tensor]:
        x, edge_index, batch, edge_feats = (
            data["pos"],
            data["edge_index"].to(torch.long),
            data["batch"],
            data.get("edge_feats"),
        )
        node_embed, _ = self.feat_embed(x)
        for module in self.node_embed:
            node_embed = module(
                x=node_embed,
                edge_index=edge_index,
                # edge_feats=edge_feats,
                # batch=batch,
            )
        graph_embed = torch.sum(node_embed, dim=0)

        return graph_embed, node_embed
