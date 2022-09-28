from typing import Union, List, Optional

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater

from src.models.cin.code.data.complex import ComplexBatch, Complex


class SSNCollater(Collater):
    def __call__(self, batch):
        if len(batch) == 1:
            return batch[0]

        if isinstance(batch[0], Complex) or isinstance(batch[0], ComplexBatch):
            return ComplexBatch.from_complex_list(batch)
        return super(SSNCollater, self).__call__(batch)

    def collate(self, batch):
        return self(batch)


class SSNDataLoader(DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
        """

    def __init__(
            self,
            dataset: Union[Dataset, List[BaseData]],
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        if "collate_fn" not in kwargs or kwargs["collate_fn"] is None:
            kwargs["collate_fn"] = SSNCollater(follow_batch, exclude_keys)

        super(SSNDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            **kwargs,
        )
