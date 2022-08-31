import os
import random

import git
import yaml
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from torch_geometric.data import Data

from src.data.data import GlycanDataModule
from src.models.class_model import ClassModel
from src.models.sweetnet.transform import SweetNetTransformer

torch.multiprocessing.set_sharing_strategy('file_system')


class NullTransformer:
    """
    Null transformer, just adding the fields that are needed to make models running in inference when trained on
    transformed data
    """

    def __init__(self, **kwargs):
        """Store which graphs should be transformed"""
        pass

    def __call__(self, data: Data):
        """Add the _x_orig filed equal to _x field, mimicking an unchanged, transformed sample"""
        return data


transforms = {
    "cin": NullTransformer,
    "gin": NullTransformer,
    "sweetnet": SweetNetTransformer,
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
    seeds = random.sample(range(1, 100), kwargs["runs"])
    for i, seed in enumerate(seeds):
        print(f"Run {i + 1} of {kwargs['runs']} with seed {seed}")
        kwargs["seed"] = seed
        single_run(**kwargs)


def single_run(**kwargs):
    """Does a single run."""
    arch = kwargs["model"]["arch"]
    # seed_everything(kwargs["seed"])
    datamodule = GlycanDataModule(
        filename=f"/home/rjo21/Desktop/SuperSweetNet/data/pred_{kwargs['datamodule']['task']}.tsv",
        transform=transforms[arch](**kwargs["model"][arch]),
        **kwargs["datamodule"],
    )

    """logger = TensorBoardLogger(
        save_dir=folder,
        name=f"version_{version}",
        version=kwargs["seed"],
        default_hp_metric=False,
    )"""
    logger = WandbLogger(log_model='all', project="pretrain_glycans", name=arch)
    logger.experiment.config.update(kwargs)

    callbacks = [
        ModelCheckpoint(monitor="loss", save_top_k=3, save_last=True, mode="min"),
        RichModelSummary(),
        RichProgressBar(),
    ]
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=25,
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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**orig_config)
