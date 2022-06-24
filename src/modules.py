from typing import Union, Tuple

from torch import nn, Tensor, LongTensor
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GATConv, GraphMultisetTransformer
from pytorch_lightning import LightningModule
from torch_geometric.typing import Adj


class MLP(LightningModule):
    """Simple Multi-layer perceptron.

    Refer to :class:`torch.nn.Sequential` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        num_layers (int, optional): Total Number of layers. Defaults to 2.
        dropout (float, optional): Dropout ratio. Defaults to 0.2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        for i in range(num_layers - 2):
            self.mlp.add_module("hidden_linear{}".format(i), nn.Linear(hidden_dim, hidden_dim))
            self.mlp.add_module("hidden_relu{}".format(i), nn.ReLU())
            self.mlp.add_module("hidden_dropout{}".format(i), nn.Dropout(dropout))
        self.mlp.add_module("final_linear", nn.Linear(hidden_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class GraphEncoder(LightningModule):
    r"""Encoder for graphs.

    Args:
        return_nodes (bool, optional): Return node embeddings as well. Defaults to False.
    """

    def __init__(self, return_nodes: bool = False, **kwargs):
        super().__init__()
        self.feat_embed = nn.Linear(kwargs["feat_dim"], kwargs["node"]["hidden_dim"])
        self.node_embed = GINConvNet(kwargs["node"]["hidden_dim"], kwargs["node"]["num_layers"])
        self.pool = GMTNet(
            input_dim=kwargs["node"]["hidden_dim"],
            hidden_dim=kwargs["pool"]["hidden_dim"],
            output_dim=kwargs["output_dim"],
            ratio=kwargs["pool"]["ratio"],
            num_heads=kwargs["pool"]["num_heads"],
        )
        self.return_nodes = return_nodes

    def forward(
        self,
        data: Union[dict, Data],
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""Encode a graph.

        Args:
            data (Union[dict, Data]): Graph to encode. Must contain the following keys:
                - x: Node features
                - edge_index: Edge indices
                - batch: Batch indices
        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Either graph of graph+node embeddings
        """
        if not isinstance(data, dict):
            data = data.to_dict()
        x, edge_index, batch, edge_feats = (
            data["x"],
            data["edge_index"],
            data["batch"],
            data.get("edge_feats"),
        )
        feat_embed = self.feat_embed(x)
        node_embed = self.node_embed(
            x=feat_embed,
            edge_index=edge_index,
            edge_feats=edge_feats,
            batch=batch,
        )
        embed = self.pool(x=node_embed, edge_index=edge_index, batch=batch)
        if self.return_nodes:
            return embed, node_embed
        return embed


class GINConvNet(LightningModule):
    """Graph Isomorphism Network.

    Refer to :class:`torch_geometric.nn.conv.GINConv` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        num_layers (int, optional): Total number of layers. Defaults to 3.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 3, **kwargs):
        super().__init__()
        self.inp = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
        )
        mid_layers = [
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.PReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                )
            )
            for _ in range(num_layers - 2)
        ]
        self.mid_layers = nn.ModuleList(mid_layers)
        self.out = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
        )

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """"""
        x = self.inp(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out(x, edge_index)
        return x


class GATConvNet(LightningModule):
    """Graph Attention Layer.

    Refer to :class:`torch_geometric.nn.conv.GATConv` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        heads (int, optional): Number of heads for multi-head attention. Defaults to 4.
        num_layers (int, optional): Number of layers. Defaults to 4.
    """

    def __init__(
        self,
        input_dim,
        output_dim: int,
        hidden_dim: int = 32,
        heads: int = 4,
        num_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.inp = GATConv(input_dim, hidden_dim, heads, concat=False)
        self.mid_layers = ModuleList(
            [GATConv(hidden_dim, hidden_dim, heads, concat=False) for _ in range(num_layers - 2)]
        )

        self.out = GATConv(hidden_dim, output_dim, concat=False)

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """"""
        x = self.inp(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out(x, edge_index)
        return x


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
        embeds = self.pool(x, batch, edge_index)
        return F.normalize(embeds, dim=1)
