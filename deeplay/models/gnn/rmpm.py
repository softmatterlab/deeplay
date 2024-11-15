from typing import Type, Union

from deeplay import (
    DeeplayModule,
    Parallel,
    MultiLayerPerceptron,
    MessagePassingNeuralNetwork,
    Sequential,
    RecurrentGraphBlock,
    CatDictElements,
)

import torch.nn as nn


class RecurrentMessagePassingModel(DeeplayModule):
    """Recurrent Message Passing Neural Network (RMPN) model."""

    hidden_features: int
    out_features: int
    num_iter: int

    def __init__(
        self,
        hidden_features: int,
        out_features: int,
        num_iter: int,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__()

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_iter = num_iter

        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        if not isinstance(hidden_features, int):
            raise ValueError(
                f"hidden_features must be an integer, got {hidden_features}"
            )

        if hidden_features <= 0:
            raise ValueError(f"hidden_features must be positive, got {hidden_features}")

        self.encoder = Parallel(
            **{
                key: MultiLayerPerceptron(
                    in_features=None,
                    hidden_features=[],
                    out_features=hidden_features,
                    flatten_input=False,
                ).set_input_map(key)
                for key in ("x", "edge_attr")
            }
        )

        combine = CatDictElements(("x", "hidden_x"), ("edge_attr", "hidden_edge_attr"))
        backbone_layer = Sequential(
            [
                Parallel(
                    **{
                        key: MultiLayerPerceptron(
                            in_features=None,
                            hidden_features=[],
                            out_features=hidden_features,
                            flatten_input=False,
                        ).set_input_map(key)
                        for key in ("hidden_x", "hidden_edge_attr")
                    }
                ),
                MessagePassingNeuralNetwork([], hidden_features),
            ]
        )
        head = MultiLayerPerceptron(
            hidden_features,
            [],
            out_features,
            out_activation=out_activation,
            flatten_input=False,
        )

        self.backbone = RecurrentGraphBlock(
            combine=combine,
            layer=backbone_layer,
            head=head,
            hidden_features=hidden_features,
            num_iter=self.num_iter,
        )

        self.backbone.layer[1].transform.set_input_map(
            "hidden_x", "edge_index", "hidden_edge_attr"
        )
        self.backbone.layer[1].transform.set_output_map("hidden_edge_attr")

        self.backbone.layer[1].propagate.set_input_map(
            "hidden_x", "edge_index", "hidden_edge_attr"
        )
        self.backbone.layer[1].propagate.set_output_map("aggregate")

        update = MultiLayerPerceptron(None, [], hidden_features, flatten_input=False)
        update.set_input_map("aggregate")
        update.set_output_map("hidden_x")
        self.backbone.layer[1].blocks[0].replace("update", update)

        self.backbone.layer[1][..., "activation"].configure(nn.ReLU)

        self.backbone.head.set_input_map("hidden_x")
        self.backbone.head.set_output_map()

    def forward(self, x):
        x = self.encoder(x)
        x = self.backbone(x)
        return x
