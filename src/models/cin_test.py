from src.data.data import GlycanDataModule
from src.models.cin.data.utils import convert_graph_dataset_with_rings

datamodule = GlycanDataModule("/home/rjo21/Desktop/SuperSweetNet/data/pred_domain.tsv", "ssn")
train_complexes, _, _ = convert_graph_dataset_with_rings(
    datamodule.train,
    max_ring_size=6,
    include_down_adj=False,
    init_edges=False,
    init_rings=False,
    n_jobs=12
)
print("Conversion completed successfully")
