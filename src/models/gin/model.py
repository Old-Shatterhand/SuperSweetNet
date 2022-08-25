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
        mlp_hidden_dims,
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
            hidden_dims=mlp_hidden_dims,
            output_dim=num_classes,
        )
