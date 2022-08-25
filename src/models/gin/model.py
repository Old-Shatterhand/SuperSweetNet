from src.models.base_model import BaseModel
from src.models.gin.modules import MLP, GraphEncoder


class GINClassModel(BaseModel):
    def __init__(
        self,
        node_feat_dim,
        node_embed_dim,
        num_gin_layers,
        graph_embed_dim,
        num_classes,
        mlp_hidden_dim,
        num_mlp_layers,
        opt_args,
        batch_size,
    ):
        super(GINClassModel, self).__init__(num_classes, batch_size, opt_args)
        self.drug_encoder = GraphEncoder(
            node_feat_dim,
            node_embed_dim,
            num_gin_layers,
            graph_embed_dim
        )
        self.mlp = MLP(
            input_dim=graph_embed_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=num_classes,
            num_layers=num_mlp_layers,
        )
