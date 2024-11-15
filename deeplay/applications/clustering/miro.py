from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from deeplay.models import RecurrentMessagePassingModel
from deeplay.applications import Application
from deeplay.external import Optimizer, Adam

from sklearn.cluster import DBSCAN


class MIRO(Application):
    num_outputs: int
    connectivity_radius: float
    model: nn.Module
    nd_loss_weight: float
    loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        num_outputs: int = 2,
        connectivity_radius: float = 1.0,
        model: Optional[nn.Module] = None,
        nd_loss_weight: float = 10,
        loss: torch.nn.Module = torch.nn.L1Loss(),
        optimizer=None,
        **kwargs,
    ):

        self.num_outputs = num_outputs
        self.connectivity_radius = connectivity_radius
        self.model = model or self._get_default_model()
        self.nd_loss_weight = nd_loss_weight

        super().__init__(loss=loss, optimizer=optimizer or Adam(lr=1e-4), **kwargs)

    def _get_default_model(self):
        rgnn = RecurrentMessagePassingModel(
            hidden_features=256, out_features=self.num_outputs, num_iter=20
        )
        return rgnn

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y_hat, y, edges, position):
        loss = 0
        for pred in y_hat:
            loss += self.loss(pred, y) + self.nd_loss_weight * self.compute_nd_loss(
                pred, y, edges, position
            )
        return loss / len(y_hat)

    def compute_nd_loss(self, y_hat, y, edges, position):
        compressed_gt = position - y * self.connectivity_radius
        compressed_gt_distances = torch.norm(
            compressed_gt[edges[0]] - compressed_gt[edges[1]], dim=1
        )
        compressed_pred = position - y_hat * self.connectivity_radius
        compressed_pred_distances = torch.norm(
            compressed_pred[edges[0]] - compressed_pred[edges[1]], dim=1
        )
        return self.loss(compressed_pred_distances, compressed_gt_distances)

    def squeeze(self, x, from_iter=-1, scaling=np.array([1.0, 1.0])):
        pred = self(x)[from_iter].detach().cpu().numpy()
        return (x.position.cpu() - pred * self.connectivity_radius).numpy() * scaling

    def clustering(
        self,
        x,
        eps,
        min_samples,
        from_iter=-1,
        **kwargs,
    ):
        squeezed = self.squeeze(x, from_iter, **kwargs)
        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(squeezed)

        return clusters.labels_

    def training_step(self, batch, batch_idx):
        x, y = self.train_preprocess(batch)
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y, x.edge_index, x.position)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log_metrics(
            "train",
            y_hat,
            y,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
