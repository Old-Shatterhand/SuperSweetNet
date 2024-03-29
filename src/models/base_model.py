from typing import Tuple, List

import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Adam, SGD, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch import nn
from torch_geometric.data import Data
from torchmetrics import Accuracy, MatthewsCorrCoef, ConfusionMatrix
import plotly.express as px

from src.models.metrics import EmbeddingMetric, MCMLAccuracy
from src.models.lr_schedules.LWCA import LinearWarmupCosineAnnealingLR
from src.models.lr_schedules.LWCAWR import LinearWarmupCosineAnnealingWarmRestartsLR


optimizers = {
    "adamw": AdamW,
    "adam": Adam,
    "sgd": SGD,
    "rmsprop": RMSprop,
}


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
        self.hidden_dims = hidden_dims

        if len(self.hidden_dims) == 0:
            self.mlp = nn.Linear(input_dim, output_dim)
        else:
            self.mlp = nn.Sequential(nn.Linear(input_dim, self.hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout))
            for i in range(len(self.hidden_dims) - 1):
                self.mlp.add_module("hidden_linear{}".format(i), nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
                self.mlp.add_module("hidden_lrelu{}".format(i), nn.LeakyReLU())
                self.mlp.add_module("batch_norm{}".format(i), nn.BatchNorm1d(self.hidden_dims[i + 1]))
                self.mlp.add_module("hidden_dropout{}".format(i), nn.Dropout(dropout))
            self.mlp.add_module("final_layer", nn.Linear(self.hidden_dims[-1], output_dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if len(self.hidden_dims) != 0:
            modules = list(self.mlp.children())
            for module in modules[:-1]:
                x = module(x)
            return modules[-1](x), x
        return self.mlp(x), x


class BaseModel(LightningModule):
    def __init__(self, classes, batch_size, opt_args):
        super().__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.batch_size = batch_size
        self.opt_config = opt_args
        self.acc = MCMLAccuracy()
        # self.mcc = MatthewsCorrCoef(num_classes=self.num_classes)
        self.confmat = ConfusionMatrix(num_classes=self.num_classes)  # , normalize='true')
        self.gnn_embed = EmbeddingMetric(classes=classes)
        self.mlp_embed = EmbeddingMetric(classes=classes)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, drug: Data) -> dict:
        """
        Forward some data through the network
        :param drug:
        :return:
        """
        drug_embed, node_embeds = self.drug_encoder(drug)
        pred, mlp_embed = self.mlp(drug_embed)
        return dict(
            drug_embed=drug_embed,
            node_embeds=node_embeds,
            pred=pred,
            mlp_embed=mlp_embed,
        )

    def training_step(self, data: Data) -> dict:
        """Step that is the same for train, validation and test.

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        fwd_dict = self.forward(data)
        labels = data.y
        ce_loss = self.loss_fn(fwd_dict["pred"], labels.float())

        self.acc.update(fwd_dict["pred"].detach(), labels.detach())
        # self.mcc.update(fwd_dict["pred"].detach(), labels.detach())

        if self.global_step % 100 == 0:
            self.confmat.update(fwd_dict["pred"].detach().argmax(dim=1), labels.detach().argmax(dim=1))
            self.gnn_embed.update(fwd_dict["drug_embed"].detach(), labels.detach().argmax(dim=1))
            self.mlp_embed.update(fwd_dict["mlp_embed"].detach(), labels.detach().argmax(dim=1))

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
        # self.log("matthews", self.mcc.compute())
        # self.mcc.reset()
        if self.global_step % 100 == 0:
            wandb.log(dict(
                confusion_matrix=self.log_confmat(),
                gnn_embeddings_tsne=self.gnn_embed.compute(),
                mlp_embeddings_tsne=self.mlp_embed.compute(),
            ))
            self.gnn_embed.reset()
            self.mlp_embed.reset()
            plt.clf()

    def log_confmat(self):
        """Log confusion matrix to wandb."""
        confmat_df = self.confmat.compute().detach().cpu().numpy()
        self.confmat.reset()

        confmat_df = pd.DataFrame(confmat_df, index=self.classes, columns=self.classes).round(2)
        return px.imshow(
            confmat_df,
            zmin=0,
            zmax=confmat_df.max().max(),
            text_auto=True,
            width=400,
            height=400,
            color_continuous_scale=px.colors.sequential.Viridis,
        )

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimizer and/or lr schedulers"""
        optimizer = optimizers[self.opt_config["module"]](params=self.parameters(), lr=self.opt_config["lr"], weight_decay=self.opt_config["weight_decay"])

        return [optimizer], [self.parse_lr_scheduler(optimizer, self.opt_config, self.opt_config["lr_schedule"])]

    def parse_lr_scheduler(self, optimizer, opt_params, lr_params):
        """Parse learning rate scheduling based on config args"""
        lr_scheduler = {"monitor": "loss"}
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
                T_max=float(opt_params["lr"]),
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
