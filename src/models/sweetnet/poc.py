from glycowork.ml.model_training import train_model, SAM
from glycowork.ml.models import prep_model
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.data import GlycanDataModule
from src.models.sweetnet.transform import SweetNetTransformer

model = prep_model("SweetNet", 4)
datamodule = GlycanDataModule(
    exp_name="ssn",
    filename=f"/home/rjo21/Desktop/SuperSweetNet/data/pred_domain.tsv",
    transform=SweetNetTransformer(),
    batch_size=128,
    num_workers=16,
)
criterion = CrossEntropyLoss()
optimizer = SAM(model.parameters(), Adam, lr=0.0005, weight_decay=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

train_model(
    model,
    {
        "train": datamodule.train_dataloader(),
        "val": datamodule.train_dataloader(),
    },
    criterion,
    optimizer,
    scheduler,
    num_epochs=150,
)
