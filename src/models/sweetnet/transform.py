import torch
from glycowork.glycowork import lib
from glycowork.motif.graph import glycan_to_graph
from torch_geometric.data import Data


class SweetNetTransformer:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data: Data) -> Data:
        a, b = glycan_to_graph(data["iupac"])
        a = [lib.index(x) if x in lib else len(lib) for x in a]
        data["x"] = torch.tensor(a)
        data["edge_index"] = torch.tensor(b)
        return data
