import torch
from glycowork.glycowork import lib
from glycowork.motif.graph import glycan_to_graph
from torch_geometric.data import Data


class SweetNetTransformer:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data: Data) -> Data:
        a, b = glycan_to_graph(data["iupac"])
        data["x"] = torch.tensor(a)
        data["edge_index"] = torch.tensor(b).long()
        return data
