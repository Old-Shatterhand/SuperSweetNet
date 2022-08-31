from typing import Any

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchmetrics import Metric
import plotly.express as px


class EmbeddingMetric(Metric):
    def __init__(self, classes):
        super(EmbeddingMetric, self).__init__()
        self.classes = classes
        self.add_state("embeds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, embeddings, labels) -> None:
        self.embeds += embeddings
        self.labels += labels

    def compute(self) -> Any:
        self.embeds = torch.stack(self.embeds).cpu().numpy()
        self.labels = torch.stack(self.labels).cpu().numpy()
        tsne_embeds = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50).fit_transform(
            self.embeds
        )
        for i, c in enumerate(self.classes):
            plt.scatter(tsne_embeds[self.labels == i, 0], tsne_embeds[self.labels == i, 1], label=c, s=20)
        embed_df = pd.DataFrame(tsne_embeds)
        embed_df["size"] = 50
        embed_df["type"] = list(map(lambda x: self.classes[x], self.labels))
        return px.scatter(
            embed_df,
            0,
            1,
            color="type",
            symbol="type",
            opacity=0.5,
            width=400,
            height=400,
            size="size",
            title="Embedding tSNE",
        )
