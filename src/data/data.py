import os
from typing import Union, List, Tuple, Callable

import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
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


class GlycanDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test."""

    def __init__(
            self,
            filename: str,
            exp_name: str,
            batch_size: int = 128,
            num_workers: int = 1,
            shuffle: bool = True,
            transform: Callable = None,
            **kwargs,
    ):
        super().__init__()
        self.filename = filename
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        """Load the individual datasets"""
        self.train = GlycanDataset(self.filename, self.exp_name, split="train", transform=transform).shuffle()
        self.val = GlycanDataset(self.filename, self.exp_name, split="val", transform=transform).shuffle()
        self.test = GlycanDataset(self.filename, self.exp_name, split="test", transform=transform).shuffle()

    def update_config(self, config: dict) -> None:
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train, **self._dl_kwargs(True))

    def val_dataloader(self):
        return DataLoader(self.val, **self._dl_kwargs(False))

    def test_dataloader(self):
        return DataLoader(self.test, **self._dl_kwargs(False))

    def predict_dataloader(self):
        return DataLoader(self.test, **self._dl_kwargs(False))

    def _dl_kwargs(self, shuffle: bool = False):
        return dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle if shuffle else False,
            num_workers=self.num_workers,
            follow_batch=["x"],
        )


class GlycanDataset(InMemoryDataset):
    def __init__(
            self,
            filename: str,
            exp_name: str,
            split: str = "train",
            transform=None,
    ):
        root = self._set_filenames(filename, exp_name)
        self.splits = {"train": 0, "val": 1, "test": 2}
        super().__init__(root, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[self.splits[split]])

    def _set_filenames(self, filename: str, exp_name: str) -> str:
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        self.filename = filename
        return os.path.join("data", exp_name, basefilename)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """Files that are created."""
        return [k + ".pt" for k in self.splits.keys()]

    def process(self):
        with open(self.filename, "r") as data_input:
            data_list = {k: [] for k in self.splits.keys()}
            for line in data_input.readlines()[1:]:
                parts = line.strip().split("\t")
                iupac, level, smiles, split = parts[0:4]
                y = torch.tensor([int(float(x)) for x in parts[4:]])
                x, edge_index = get_graph(smiles)
                if x is None and edge_index is None:
                    continue
                data_list[split].append(Data(x=x, edge_index=edge_index, y=y.unsqueeze(0), iupac=iupac, level=level))
            for split in self.splits.keys():
                data, slices = self.collate(data_list[split])
                torch.save((data, slices), self.processed_paths[self.splits[split]])


def encode_atom(atom):
    if atom.GetAtomicNum() in encodings["glycan"]:
        return encodings["glycan"][atom.GetAtomicNum()] + encodings["chirality"][atom.GetChiralTag()]
    else:
        return encodings["glycan"]["other"] + encodings["chirality"][atom.GetChiralTag()]


def get_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, None
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    edges = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([start, end])
        edges.append([end, start])
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(encode_atom(atom))
    return torch.tensor(atom_features, dtype=torch.float), torch.tensor(edges, dtype=torch.long).T
