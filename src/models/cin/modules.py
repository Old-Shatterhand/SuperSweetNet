from pytorch_lightning import LightningModule

from src.models.cin.code.mp.models import CIN0


class SimplexEncoder(LightningModule):
    def __init__(
            self,
            graph_embed_dim,
            node_feat_dim,
            node_embed_dim,
            num_gnn_layers,
            **kwargs,
    ):
        super(SimplexEncoder, self).__init__()
        self.module = CIN0(
            node_feat_dim,
            graph_embed_dim,
            num_gnn_layers,
            node_embed_dim,
        )

    def forward(self, data):
        return self.module.forward(data), None
