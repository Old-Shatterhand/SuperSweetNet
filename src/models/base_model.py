from typing import Tuple, List

import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Adam, SGD, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch import nn
from torch_geometric.data import Data
from torchmetrics import Accuracy, MatthewsCorrCoef, ConfusionMatrix
import plotly.express as px

from src.models.metrics import EmbeddingMetric
from src.models.lr_schedules.LWCA import LinearWarmupCosineAnnealingLR
from src.models.lr_schedules.LWCAWR import LinearWarmupCosineAnnealingWarmRestartsLR


class MLP(LightningModule):
    """Simple Multi-layer perceptron.

    Refer to :class:`torch.nn.Sequential` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        num_layers (int, optional): Total Number of layers. Defaults to 2.
        dropout (float, optional): Dropout ratio. Defaults to 0.2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        if len(hidden_dims) == 0:
            self.mlp = nn.Linear(input_dim, output_dim)
        else:
            self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout))
            for i in range(len(hidden_dims) - 1):
                self.mlp.add_module("hidden_linear{}".format(i), nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                self.mlp.add_module("hidden_relu{}".format(i), nn.ReLU())
                self.mlp.add_module("hidden_dropout{}".format(i), nn.Dropout(dropout))
            self.mlp.add_module("final_layer", nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class BaseModel(LightningModule):
    def __init__(self, classes, batch_size, opt_args):
        super().__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.batch_size = batch_size
        self.opt_config = opt_args
        self.acc = Accuracy(num_classes=self.num_classes)
        self.mcc = MatthewsCorrCoef(num_classes=self.num_classes)
        self.confmat = ConfusionMatrix(num_classes=self.num_classes, normalize='true')
        self.embed = EmbeddingMetric(classes=classes)

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

    def training_step(self, data: Data) -> dict:
        """Step that is the same for train, validation and test.

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        fwd_dict = self.forward(data)
        labels = data.y
        ce_loss = F.cross_entropy(fwd_dict["pred"], labels.float().argmax(dim=1))

        self.acc.update(fwd_dict["pred"].detach().argmax(dim=1), labels.detach().argmax(dim=1))
        self.mcc.update(fwd_dict["pred"].detach().argmax(dim=1), labels.detach().argmax(dim=1))

        if self.global_step % 100 == 0:
            self.confmat.update(fwd_dict["pred"].detach().argmax(dim=1), labels.detach().argmax(dim=1))
            self.embed.update(fwd_dict["drug_embed"].detach(), labels.detach().argmax(dim=1))

        self.log("loss", ce_loss, batch_size=self.batch_size)

        output = dict(
            loss=ce_loss,
            preds=fwd_dict["pred"].detach(),
            labels=labels.detach(),
            drug_embed=fwd_dict["drug_embed"].detach(),
        )
        if fwd_dict["node_embeds"] is not None:
            output["node_embeds"] = fwd_dict["node_embeds"].detach()
        return output

    def training_epoch_end(self, outputs: dict):
        """What to do at the end of a training epoch. Logs everything."""
        self.log("accuracy", self.acc.compute())
        self.acc.reset()
        self.log("matthews", self.mcc.compute())
        self.mcc.reset()
        if self.global_step % 100 == 0:
            wandb.log(dict(
                confusion_matrix=self.log_confmat(),
                embeddings_tsne=self.embed.compute(),
            ))
            self.embed.reset()
            plt.clf()

    def log_confmat(self):
        """Log confusion matrix to wandb."""
        confmat_df = self.confmat.compute().detach().cpu().numpy()
        self.confmat.reset()

        confmat_df = pd.DataFrame(confmat_df, index=self.classes, columns=self.classes).round(2)
        return px.imshow(
            confmat_df,
            zmin=0,
            zmax=1,
            text_auto=True,
            width=400,
            height=400,
            color_continuous_scale=px.colors.sequential.Viridis,
        )

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
        return [optimizer], [self.parse_lr_scheduler(optimizer, opt_params, opt_params["lr_schedule"])]

    def parse_lr_scheduler(self, optimizer, opt_params, lr_params):
        """Parse learning rate scheduling based on config args"""
        lr_scheduler = {"monitor": self.hparams["early_stop"]["monitor"]}
        if lr_params["module"] == "rlrop":
            lr_scheduler["scheduler"] = ReduceLROnPlateau(
                optimizer,
                verbose=True,
                factor=lr_params["factor"],
                patience=lr_params["patience"],
            )
        elif lr_params["module"] == "calr":
            lr_scheduler["scheduler"] = CosineAnnealingLR(
                optimizer,
                peak_lr=float(opt_params["lr"]),
            )
        elif lr_params["module"] == "lwcawr":
            lr_scheduler["scheduler"] = LinearWarmupCosineAnnealingWarmRestartsLR(
                optimizer,
                warmup_epochs=lr_params["warmup_epochs"],
                start_lr=float(lr_params["start_lr"]),
                peak_lr=float(opt_params["lr"]),
                cos_restart_dist=lr_params["cos_restart_dist"],
                cos_eta_min=float(lr_params["min_lr"])
            )
        elif lr_params["module"] == "lwca":
            lr_scheduler["scheduler"] = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=lr_params["warmup_epochs"],
                max_epochs=lr_params["cos_restart_dist"],
                eta_min=float(lr_params["min_lr"]),
                warmup_start_lr=float(lr_params["start_lr"]),
            )
        else:
            raise ValueError("Unknown learning rate scheduler")

        return lr_scheduler
