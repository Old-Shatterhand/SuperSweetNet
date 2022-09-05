import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit.Chem.rdchem import ChiralType


encodings = {
    "glycan": {
        "other": [0, 0, 0],
        6: [1, 0, 0],  # carbon
        7: [0, 1, 0],  # nitrogen
        8: [0, 0, 1],  # oxygen
    },
    "chirality": {
        ChiralType.CHI_OTHER: [0, 0, 0],
        ChiralType.CHI_TETRAHEDRAL_CCW: [
            1,
            1,
            0,
        ],  # counterclockwise rotation of polarized light -> rotate light to the left
        ChiralType.CHI_TETRAHEDRAL_CW: [1, 0, 1],  # clockwise rotation of polarized light -> rotate light to the right
        ChiralType.CHI_UNSPECIFIED: [0, 0, 0],
    },
}


class SMILESTransformer:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data: Data) -> Data:
        x, edge_index = get_graph(data["smiles"])
        data["x"] = x
        data["edge_index"] = edge_index
        return data


def encode_atom(atom):
    if atom.GetAtomicNum() in encodings["glycan"]:
        return encodings["glycan"][atom.GetAtomicNum()] + encodings["chirality"][atom.GetChiralTag()]
    else:
        return encodings["glycan"]["other"] + encodings["chirality"][atom.GetChiralTag()]


def get_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("mol is none")
        return None, None
    # new_order = rdmolfiles.CanonicalRankAtoms(mol)
    # mol = rdmolops.RenumberAtoms(mol, new_order)
    edges = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([start, end])
        edges.append([end, start])
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(encode_atom(atom))
    return torch.tensor(atom_features, dtype=torch.float), torch.tensor(edges, dtype=torch.long).T
