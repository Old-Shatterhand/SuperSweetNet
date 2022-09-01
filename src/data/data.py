import copy
import os
from typing import Union, List, Tuple, Callable

from rdkit import Chem
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader


class GlycanDataModule(LightningDataModule):
    """Base data module, contains all the datasets for train, val and test."""

    def __init__(
            self,
            filename: str,
            exp_name: str,
            batch_size: int = 128,
            num_workers: int = 1,
            shuffle: bool = True,
            pre_transform: Callable = None,
            init_filter: Callable = None,
            init_transform: Callable = None,
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
        self.train = GlycanDataset(self.filename, self.exp_name, pre_transform=pre_transform, init_filter=init_filter,
                                   init_transform=init_transform, transform=transform).shuffle()
        # self.val = GlycanDataset(self.filename, self.exp_name, transform=transform).shuffle()
        # self.test = GlycanDataset(self.filename, self.exp_name, transform=transform).shuffle()

    def update_config(self, config: dict) -> None:
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train, **self._dl_kwargs(True))

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
            pre_transform: Callable = None,
            init_filter: Callable = None,
            init_transform: Callable = None,
            transform=None,
    ):
        self.loaded = False
        root = self._set_filenames(filename, exp_name)
        super(GlycanDataset, self).__init__(root, pre_transform=pre_transform, transform=transform)
        self.classes = None

        if init_filter is not None or init_transform is not None:
            self.unsaved_preps(init_filter, init_transform)
        else:
            self.data, self.slices, self.classes = torch.load(self.processed_paths[0])

    def unsaved_preps(self, init_filter, init_transform):
        tmp_data, tmp_slices, self.classes = torch.load(self.processed_paths[0])

        def get(idx):
            return separate(
                cls=tmp_data.__class__,
                batch=tmp_data,
                idx=idx,
                slice_dict=tmp_slices,
                decrement=False,
            )

        if init_filter is not None:
            datalist = list(filter(init_filter, [get(i) for i in range(tmp_data["y"].size(0))]))
            if init_transform is not None:
                datalist = [init_transform(d) for d in datalist]
        elif init_transform is not None:
            datalist = [init_transform(d) for d in [get(i) for i in range(tmp_data["y"].size(0))]]
        if init_filter is not None or init_transform is not None:
            # print(all([Chem.MolFromSmiles(d["smiles"]) is not None for d in datalist]))
            self.data, self.slices = self.collate(datalist)
            # print(all([Chem.MolFromSmiles(d) is not None for d in self.data["smiles"]]))

        self.loaded = True
        # print(all([Chem.MolFromSmiles(self[i]["smiles"]) is not None for i in range(len(self.data))]))
        # print(all([Chem.MolFromSmiles(d["smiles"]) is not None for d in self]))
        print("Initializing done")

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
        with open(self.filename, "r") as data_input:
            data_list = []
            classes = []
            for i, line in enumerate(data_input.readlines()):
                if i == 0:
                    classes = line.strip().split("\t")[4:]
                else:
                    parts = line.strip().split("\t")
                    iupac, level, smiles, split = parts[0:4]
                    y = torch.tensor([int(float(x)) for x in parts[4:]]).unsqueeze(0)
                    data_list.append(Data(x=None, edge_index=None, y=y, iupac=iupac, level=level, smiles=smiles))
            data, slices = self.collate(data_list)
            torch.save((data, slices, classes), self.processed_paths[0])
