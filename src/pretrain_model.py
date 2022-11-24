from typing import Tuple

import pandas as pd
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import wandb
from torch_geometric.data import Data
from torchmetrics import ConfusionMatrix, Accuracy, MeanSquaredError
from torchmetrics.functional.classification import accuracy
import plotly.express as px
from src.models.base_model import BaseModel, MLP, optimizers
from src.models.lr_schedules.LWCA import LinearWarmupCosineAnnealingLR
from src.models.lr_schedules.LWCAWR import LinearWarmupCosineAnnealingWarmRestartsLR
from src.models.pretrain.modules import PositionalEncoder

encoders = {
    "pos": PositionalEncoder,
}


class PretrainModel(LightningModule):
    def __init__(
            self,
            graph_embed_dim,
            encoder,
            encoder_args,
            mlp_hidden_dims,
            atom_types,
            batch_size,
            opt_args,
    ):
        super(PretrainModel, self).__init__()
        self.drug_encoder = encoders[encoder](graph_embed_dim, **encoder_args)
        self.pos_mlp = MLP(
            input_dim=graph_embed_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=3,
        )
        self.node_mlp = MLP(
            input_dim=graph_embed_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=len(atom_types),
        )

        self.alpha = encoder_args.get("alpha", 4)
        self.confmat = ConfusionMatrix(num_classes=len(atom_types), normalize="true")
        self.acc = Accuracy(num_classes=len(atom_types))
        self.mse = MeanSquaredError()
        self.opt_config = opt_args
        self.atom_types = atom_types

    def forward(self, data: Data) -> dict:
        drug_embed, node_embeds = self.drug_encoder(data)
        noise_preds = self.pos_mlp(node_embeds)
        node_preds = self.node_mlp(node_embeds)
        node_preds = torch.nn.functional.softmax(node_preds[0]), node_preds[1]
        return dict(
            drug_embed=drug_embed,
            node_embeds=node_embeds,
            noise_pred=noise_preds[0],
            node_pred=node_preds[0],
        )

    def training_step(self, data: Data, batch_idx, **kwargs):
        fwd_dict = self.forward(data)

        noise_loss = F.mse_loss(fwd_dict["noise_pred"], data.noise)
        label_loss = F.cross_entropy(fwd_dict["node_pred"][data.mask], data.orig_x[data.mask])
        loss = noise_loss + self.alpha * label_loss

        self.acc.update(fwd_dict["node_pred"][data.mask], data.orig_x[data.mask])
        self.mse.update(fwd_dict["noise_pred"], data.noise)
        self.confmat.update(fwd_dict["node_pred"][data.mask], data.orig_x[data.mask])

        self.log("loss", loss)
        self.log("noise_loss", noise_loss)
        self.log("label_loss", label_loss)

        return dict(
            loss=loss,
            noise_loss=noise_loss.detach(),
            label_loss=label_loss.detach(),
        )

    def training_epoch_end(self, outputs: dict):
        self.log("accuracy", self.acc.compute())
        self.acc.reset()
        self.log("distance", self.mse.compute())
        self.mse.reset()
        if self.global_step % 10 == 0:
            wandb.log(dict(
                confusion_matrix=self.log_confmat()
            ))

    def log_confmat(self):
        """Log confusion matrix to wandb."""
        confmat_df = self.confmat.compute().detach().cpu().numpy()
        self.confmat.reset()

        confmat_df = pd.DataFrame(confmat_df, index=self.atom_types, columns=self.atom_types).round(2)
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
