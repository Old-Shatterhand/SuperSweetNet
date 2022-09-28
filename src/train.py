import os
import random

import git
import wandb
import yaml
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

from src.data.data import GlycanDataModule
from src.models.cin.transform import CINTransformer
from src.models.class_model import ClassModel
from src.models.gin.transform import SMILESTransformer
from src.models.sweetnet.transform import SweetNetTransformer
from src.utils import IterDict

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["WANDB_CACHE_DIR"] = "/scratch/SCRATCH_SAS/roman/.config/wandb"


init_filters = {
    "cin": lambda x: len(x["smiles"]) > 10 and Chem.MolFromSmiles(x["smiles"]) is not None,
    "gin": lambda x: len(x["smiles"]) > 10 and Chem.MolFromSmiles(x["smiles"]) is not None,
    "sweetnet": None,
}


init_transforms = {
    "cin": Compose([SMILESTransformer(), CINTransformer()]),
    "gin": SMILESTransformer(),
    "sweetnet": SweetNetTransformer(),
}


def read_config(filename: str) -> dict:
    """Read in yaml config for training."""
    with open(filename, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def get_git_hash():
    """Get the git hash of the current repository."""
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def train(**kwargs):
    seed_everything(42)
    seeds = random.sample(range(1, 100), kwargs["runs"])
    for i, seed in enumerate(seeds):
        print(f"Run {i + 1} of {kwargs['runs']} with seed {seed}")
        kwargs["seed"] = seed
        single_run(**kwargs)


def single_run(**kwargs):
    """Does a single run."""
    arch = kwargs["model"]["arch"]
    seed_everything(kwargs["seed"])
    datamodule = GlycanDataModule(
        filename=f"/home/rjo21/Desktop/SuperSweetNet/data/pred_{kwargs['datamodule']['task']}.tsv",
        # filename=f"data/pred_{kwargs['datamodule']['task']}_2.tsv",
        init_filter=init_filters[arch],
        init_transform=init_transforms[arch],  #  (**kwargs["model"][arch]),
        **kwargs["datamodule"],
    )
    # logger = WandbLogger(
    #     log_model='all',
    #     project="pretrain_glycans",
    #     name=f"{arch}_{kwargs['model']['postfix']}_{kwargs['datamodule']['task']}"
    # )
    # logger.experiment.config.update(kwargs)

    callbacks = [
        ModelCheckpoint(monitor="loss", save_top_k=3, save_last=True, mode="min"),
        EarlyStopping(
            monitor=kwargs["early_stop"]["monitor"],
            mode=kwargs["early_stop"]["mode"],
            patience=kwargs["early_stop"]["patience"]
        ),
        RichModelSummary(),
        RichProgressBar(),
    ]
    trainer = Trainer(
        callbacks=callbacks,
        # logger=logger,
        log_every_n_steps=25,
        # limit_train_batches=10,
        enable_model_summary=False,
        **kwargs["trainer"],
    )
    print(datamodule.train.classes)
    model = ClassModel(
        kwargs["model"]["graph_embed_dim"],
        arch,
        kwargs["model"][arch],
        kwargs["model"]["hidden_dims"],
        datamodule.train.classes,
        kwargs["datamodule"]["batch_size"],
        kwargs["optimizer"],
    )

    print("Model buildup finished")
    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule)
    # wandb.finish()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    for task in [
        "domain",
        # "kingdom",
        # "phylum",
        # "class",
        # "order",
        # "genus",
        # "species"
    ]:
        orig_config["datamodule"]["task"] = task
        train(**orig_config)
