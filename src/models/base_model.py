from typing import Tuple

import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torchmetrics import MetricCollection, Accuracy, MatthewsCorrCoef

from src.models.gin.modules import MLP, GraphEncoder


class BaseModel(LightningModule):
    def __init__(self, num_classes, batch_size, opt_args):
        super().__init__()
        self._set_class_metrics(num_classes)
        self.batch_size = batch_size
        self.opt_config = opt_args

    def _set_class_metrics(self, num_classes: int):
        metrics = MetricCollection(
            [
                Accuracy(num_classes=None if num_classes == 2 else num_classes),
                MatthewsCorrCoef(num_classes=num_classes),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, drug: Data) -> dict:
        """
        Forward some data through the network
        :param drug:
        :return:
        """
        drug_embed, node_embeds = self.drug_encoder(drug)
        return dict(
            drug_embed=drug_embed,
            node_embeds=node_embeds,
            pred=self.mlp(drug_embed),
        )

    def shared_step(self, data: Data) -> dict:
        """Step that is the same for train, validation and test.

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        fwd_dict = self.forward(data)
        labels = data.y
        ce_loss = F.cross_entropy(fwd_dict["pred"], labels.float().argmax(dim=1))
        return dict(loss=ce_loss, preds=fwd_dict["pred"].detach(), labels=labels.detach())

    def training_step(self, data: Data, data_idx: int) -> dict:
        """What to do during training step."""
        ss = self.shared_step(data)
        self.train_metrics.update(ss["preds"].argmax(dim=1), ss["labels"].argmax(dim=1))
        self.log("train_loss", ss["loss"], batch_size=self.batch_size)
        return ss

    def validation_step(self, data: Data, data_idx: int) -> dict:
        """What to do during validation step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        self.val_metrics.update(ss["preds"].argmax(dim=1), ss["labels"].argmax(dim=1))
        self.log("val_loss", ss["loss"], batch_size=self.batch_size)
        return ss

    def test_step(self, data: Data, data_idx: int) -> dict:
        """What to do during test step. Also logs the values for various callbacks."""
        ss = self.shared_step(data)
        self.test_metrics.update(ss["preds"].argmax(dim=1), ss["labels"].argmax(dim=1))
        self.log("test_loss", ss["loss"], batch_size=self.batch_size)
        return ss

    def log_histograms(self):
        """Logs the histograms of all the available parameters."""
        if self.logger:
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def log_all(self, metrics: dict, hparams: bool = False):
        """Log all metrics."""
        if self.logger:
            for k, v in metrics.items():
                self.logger.experiment.add_scalar(k, v, self.current_epoch)
            if hparams:
                self.logger.log_hyperparams(self.hparams, {k.split("_")[-1]: v for k, v in metrics.items()})

    def training_epoch_end(self, outputs: dict):
        """What to do at the end of a training epoch. Logs everything."""
        self.log_histograms()
        metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        self.log_all(metrics)

    def validation_epoch_end(self, outputs: dict):
        """What to do at the end of a validation epoch. Logs everything."""
        metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log_all(metrics, hparams=True)

    def test_epoch_end(self, outputs: dict):
        """What to do at the end of a test epoch. Logs everything."""
        metrics = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_all(metrics)

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimizer and/or lr schedulers"""
        opt_class = {"adamw": AdamW, "adam": Adam, "sgd": SGD, "rmsprop": RMSprop}[self.opt_config["module"]]
        opt_params = {
            "params": self.parameters(),
            "lr": self.opt_config["lr"],
            "weight_decay": self.opt_config["weight_decay"],
        }
        if self.opt_config["module"] in ["sgd", "rmsprop"]:  # not adam or adamw
            opt_params["momentum"] = self.opt_config["momentum"]
        optimizer = opt_class(**opt_params)

        lr_scheduler = {
            "monitor": self.opt_config["reduce_lr"]["monitor"],
            "scheduler": ReduceLROnPlateau(
                optimizer,
                verbose=True,
                factor=self.opt_config["reduce_lr"]["factor"],
                patience=self.opt_config["reduce_lr"]["patience"],
            ),
        }
        return [optimizer], [lr_scheduler]
