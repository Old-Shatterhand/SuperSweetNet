import os
from typing import Callable, Union, List, Tuple

import torch
from pytorch_lightning import LightningDataModule
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeRotation


class PretrainDataModule(LightningDataModule):
    def __init__(
            self,
            filename: str,
            exp_name: str,
            batch_size: int = 128,
            num_workers: int = 1,
            shuffle: bool = True,
            pre_transform: Callable = None,
            transform: Callable = None,
            **kwargs,
    ):
        super().__init__()
        self.filename = filename
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train = PretrainDataset(self.filename, self.exp_name, pre_transform=pre_transform, transform=transform).shuffle()

    def update_config(self, config: dict) -> None:
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    def predict_dataloader(self):
        raise NotImplementedError()


class PretrainDataset(InMemoryDataset):
    def __init__(
            self,
            filename: str,
            exp_name: str,
            pre_transform: Callable = None,
            transform: Callable = None,
    ):
        root = self._set_filenames(filename, exp_name)
        super(PretrainDataset, self).__init__(root, pre_transform=pre_transform, transform=transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def _set_filenames(self, filename: str, exp_name: str) -> str:
        basefilename = os.path.basename(filename)
        basefilename = os.path.splitext(basefilename)[0]
        self.filename = filename
        return os.path.join("data", exp_name, basefilename)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """Files that are created."""
        return ['train.pt']

    def process(self):
        data_list = []

        for pdb in os.listdir(self.filename):
            if not pdb.endswith(".pdb"):
                return
            mol = Chem.MolFromPDBFile(os.path.join(self.filename, pdb))
            if mol is None:
                return
            data = mol_to_pyg(mol)
            data_list.append(NormalizeRotation()(data))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# ["C", "N", "O", "S", "R", "M"]
atom_map = {
    6: 0,
    7: 1,
    8: 2,
    16: 3,
    "other": 4,
    "mask": 5,
}


def mol_to_pyg(mol):
    conf = mol.GetConformer()
    pos = []
    x = []
    for i, atom in enumerate(mol.GetAtoms()):
        positions = conf.GetAtomPosition(i)
        pos.append((positions.x, positions.y, positions.z))
        x.append(atom_map.get(atom.GetAtomicNum(), 4))
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return Data(x=torch.tensor(x), pos=torch.tensor(pos), edge_index=torch.tensor(bonds).T, y=torch.tensor(0))
