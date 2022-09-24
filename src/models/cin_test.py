import torch

from src.data.data import GlycanDataModule
from src.models.cin.code.data.utils import convert_graph_dataset_with_rings
from src.train import init_filters, init_transforms
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    datamodule = GlycanDataModule(
        "data/pred_domain.tsv",
        "ssn",
        batch_size=1,
        init_filter=init_filters["cin"],
        init_transform=init_transforms["cin"](),
    )

    train_complexes, _, _ = convert_graph_dataset_with_rings(
        datamodule.train,
        max_ring_size=6,
        include_down_adj=True,
        init_edges=True,
        init_rings=True,
        n_jobs=1,
    )
    print("Conversion completed successfully")
