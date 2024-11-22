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
    """Recurrent Message Passing Neural Network (RMPN) model.

    Parameters
    ----------
    hidden_features: int
        Number of hidden units in the recurrent message passing layer.
    out_features: int
        Number of output features.
    num_iter: int
        Number of iterations of the recurrent message passing layer.
    out_activation: template-like
        Specification for the output activation of the model. Default: nn.Identity.


    Configurables
    -------------
    - hidden_features (int): Number of hidden units in the recurrent message passing layer.
    - out_features (int): Number of output features.
    - out_activation (template-like): Specification for the output activation of the model. Default: nn.Identity.
    - encoder (template-like): Specification for the encoder of the model. Default: dl.Parallel consisting of two MLPs to process node and edge features.
    - backbone (template-like): Specification for the backbone of the model. Default: dl.RecurrentGraphBlock consisting of dl.MessagePassingNeuralNetwork and a MLP head.

    Constraints
    -----------
    - input: Dict[str, Any] or torch-geometric Data object containing the following attributes:
        - x: torch.Tensor of shape (num_nodes, node_in_features).
        - edge_index: torch.Tensor of shape (2, num_edges).
        - edge_attr: torch.Tensor of shape (num_edges, edge_in_features).
        - hidden_x: (Optional) torch.Tensor of shape (num_nodes, hidden_features).
        - hidden_edge_attr: (Optional) torch.Tensor of shape (num_edges, hidden_features).

        NOTE: node_in_features and edge_in_features are inferred from the input data.

    - output: List[torch.Tensor] where each tensor has shape (num_nodes, out_features).

    Examples
    --------
    >>> # Define a RMPN model with 96 hidden features, 2 output features, and 3 iterations
    >>> model = RecurrentMessagePassingModel(hidden_features=96, out_features=2, num_iter=3)

    >>> # Input graph data
    >>> inp = {
    >>>     "x": torch.randn(10, 5),  # Node features
    >>>     "edge_index": torch.randint(0, 10, (2, 20)),  # Edge connectivity
    >>>     "edge_attr": torch.randn(20, 3),  # Edge features
    >>> }

    >>> # Model forward pass
    >>> out = model(inp)

    >>> # Output shape
    >>> print(len(out))
    3
    >>> print(out[0].shape)
    torch.Size([10, 2])
    """

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
