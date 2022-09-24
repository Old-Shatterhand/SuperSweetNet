import torch
from torch import Tensor

from src.models.cin.code.data.utils import compute_ring_2complex


def maybe_convert_to_numpy(x):
    if isinstance(x, Tensor):
        return x.numpy()
    return x


class CINTransformer:
    def __init__(
            self,
            max_ring_size=7,
            include_down_adj=False,
            init_method: str = 'sum',
            init_edges=True,
            init_rings=True,
            **kwargs
    ):
        self.max_ring_size = max_ring_size
        self.include_down_adj = include_down_adj
        self.init_method = init_method
        self.init_edges = init_edges
        self.init_rings = init_rings

    def __call__(self, data):
        return compute_ring_2complex(
            maybe_convert_to_numpy(data.x),
            maybe_convert_to_numpy(data.edge_index),
            maybe_convert_to_numpy(data.edge_attr),
            data.num_nodes,
            y=maybe_convert_to_numpy(data.y),
            max_k=self.max_ring_size,
            include_down_adj=self.include_down_adj,
            init_method=self.init_method,
            init_edges=self.init_edges,
            init_rings=self.init_rings,
        )
