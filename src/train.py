import os
import random

import git
import yaml
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from src.data.data import GlycanDataModule
from src.models.gin.model import GINClassModel

torch.multiprocessing.set_sharing_strategy('file_system')

models = {
    "gin": GINClassModel,
    "cin": None,
    "sweetnet": None,
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
    seed_everything(kwargs["seed"])
    seeds = random.sample(range(1, 100), kwargs["runs"])

    folder = os.path.join(
        "tb_logs",
        f"gly_{kwargs['datamodule']['exp_name']}",
        f"{kwargs['datamodule']['filename'].split('/')[-1].split('.')[0]}",
    )
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if len(os.listdir(folder)) == 0:
        next_version = 0
    else:
        next_version = str(int([d for d in os.listdir(folder)
                                if "version" in d and os.path.isdir(os.path.join(folder, d))][-1].split("_")[1]) + 1)

    for i, seed in enumerate(seeds):
        print(f"Run {i + 1} of {kwargs['runs']} with seed {seed}")
        kwargs["seed"] = seed
        single_run(folder, next_version, **kwargs)


def single_run(folder, version, **kwargs):
    """Does a single run."""
    seed_everything(kwargs["seed"])
    datamodule = GlycanDataModule(**kwargs["datamodule"])

    logger = TensorBoardLogger(
        save_dir=folder,
        name=f"version_{version}",
        version=kwargs["seed"],
        default_hp_metric=False,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=3, save_last=True, mode="min"),
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
    model = models[kwargs["model"]["arch"]](
        num_classes=kwargs["datamodule"]["num_classes"],
        opt_args=kwargs["optimizer"],
        batch_size=kwargs["datamodule"]["batch_size"],
        **kwargs["model"][kwargs["model"]["arch"]]
    )

    print("Model buildup finished")
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    train(**orig_config)
