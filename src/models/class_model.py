from src.models.base_model import BaseModel, MLP
from src.models.gin.modules import GraphEncoder
from src.models.sweetnet.modules import SweetNetEncoder

encoders = {
    "cin": None,
    "gin": GraphEncoder,
    "sweetnet": SweetNetEncoder,
}


class ClassModel(BaseModel):
    def __init__(
            self,
            graph_embed_dim,
            encoder,
            encoder_args,
            mlp_hidden_dims,
            classes,
            batch_size,
            opt_args,
    ):
        super(ClassModel, self).__init__(classes, batch_size, opt_args)
        self.drug_encoder = encoders[encoder](graph_embed_dim, **encoder_args)
        self.mlp = MLP(
            input_dim=graph_embed_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=len(classes),
        )
