import torch
from torch_geometric.data import batch, Data
from torch_geometric.transforms import BaseTransform


class PosNoise(BaseTransform):
    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def __call__(self, t_batch) -> torch.Tensor:
        noise = torch.randn_like(t_batch.pos) * self.sigma
        t_batch.pos += noise
        t_batch.noise = noise
        return t_batch


class MaskType(BaseTransform):
    """Masks the type of the nodes in a graph."""

    def __init__(self, pick_prob: float = 0.15):
        self.prob = pick_prob

    def __call__(self, t_batch) -> torch.Tensor:
        mask = torch.rand_like(t_batch.x, dtype=torch.float32) < self.prob
        t_batch.orig_x = t_batch.x.clone()
        t_batch.x[mask] = 5
        t_batch.mask = mask
        return t_batch
