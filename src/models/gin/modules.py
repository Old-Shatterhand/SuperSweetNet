from typing import Union, Tuple, Callable

import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GraphMultisetTransformer, global_mean_pool
from pytorch_lightning import LightningModule
from torch_geometric.typing import Adj

from src.models.base_model import MLP


class GraphEncoder(LightningModule):
    r"""Encoder for graphs.

    Args:

    """

    def __init__(
            self,
            graph_embed_dim: int,
            node_feat_dim: int,
            node_embed_dim: int,
            num_gnn_layers: int,
            **kwargs,
    ):
        super().__init__()
        self.feat_embed = MLP(node_feat_dim, [64], graph_embed_dim)
        self.node_embed = GINConvNet(graph_embed_dim, global_mean_pool, num_gnn_layers)

    def forward(self, data: Union[dict, Data], **kwargs,) -> Tuple[Tensor, Tensor]:
        r"""Encode a graph.

        Args:
            data (Union[dict, Data]): Graph to encode. Must contain the following keys:
                - x: Node features
                - edge_index: Edge indices
                - batch: Batch indices
        Returns:
            dict: Either graph of graph+node embeddings
        """
        if not isinstance(data, dict):
            data = data.to_dict()
        x, edge_index, batch, edge_feats = (
            data["x"],
            data["edge_index"].to(torch.long),
            data["batch"],
            data.get("edge_feats"),
        )
        feat_embed, _ = self.feat_embed(x)
        graph_embed, node_embed = self.node_embed(
            x=feat_embed,
            edge_index=edge_index,
            edge_feats=edge_feats,
            batch=batch,
        )

        return graph_embed, node_embed


class GINConvNet(LightningModule):
    """Graph Isomorphism Network.

    Refer to :class:`torch_geometric.nn.conv.GINConv` for more details.

    Args:
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        num_layers (int, optional): Total number of layers. Defaults to 3.
    """

    def __init__(self, hidden_dim: int, pooling: Callable, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    # nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                )
            )
            for _ in range(num_layers)
        ])

        self.pooling = pooling

    def forward(self, x: Tensor, edge_index: Adj, batch, **kwargs) -> Tuple[Tensor, Tensor]:
        """"""
        pools = []
        for module in self.layers:
            x = module(x, edge_index)
            pools.append(self.pooling(x=x, batch=batch))
        return torch.stack(pools).sum(dim=0), x


class GMTNet(LightningModule):
    """Graph Multiset Transformer pooling.

    Refer to :class:`torch_geometric.nn.glob.GraphMultisetTransformer` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden layer(s). Defaults to 128.
        ratio (float, optional): Ratio of the number of nodes to be pooled. Defaults to 0.25.
        max_nodes (int, optional): Maximal number of nodes in a graph. Defaults to 600.
        num_heads (int, optional): Number of heads. Defaults to 4.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        ratio: float = 0.25,
        max_nodes: int = 600,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.pool = GraphMultisetTransformer(
            input_dim,
            hidden_dim,
            output_dim,
            num_nodes=max_nodes * 1.5,
            pooling_ratio=ratio,
            num_heads=num_heads,
            pool_sequences=["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"],
            layer_norm=True,
        )

    def forward(self, x: Tensor, edge_index: Adj, batch: LongTensor) -> Tensor:
        """"""
        embeds = self.pool(x, batch, edge_index=edge_index)
        return F.normalize(embeds, dim=1)
