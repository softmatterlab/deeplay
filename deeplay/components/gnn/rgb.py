import torch
from torch_geometric.data import Data
from deeplay import DeeplayModule
from deeplay.components.dict import CatDictElements


class RecurrentGraphBlock(DeeplayModule):
    combine: DeeplayModule
    layer: DeeplayModule
    head: DeeplayModule

    def __init__(
        self,
        layer: DeeplayModule,
        head: DeeplayModule,
        hidden_features: int,
        num_iter: int,
        combine: DeeplayModule = CatDictElements(("x", "hidden")),
    ):
        super().__init__()
        self.combine = combine
        self.layer = layer
        self.head = head
        self.hidden_features = hidden_features
        self.num_iter = num_iter

        if not all(hasattr(self.combine, attr) for attr in ("source", "target")):
            raise AttributeError(
                "The 'combine' module must have 'source' and 'target' attributes to specify "
                "the keys to concatenate. Found None. Ensure that the 'combine' module is initialized "
                "with valid 'source' and 'target' keys. Check CatDictElements for reference."
            )
        self.hidden_variables_name = self.combine.target

    def initialize_hidden(self, x):
        x = x.clone() if isinstance(x, Data) else x.copy()
        for source, hidden_variable_name in zip(
            self.combine.source, self.hidden_variables_name
        ):
            if hidden_variable_name not in x:
                x.update(
                    {
                        hidden_variable_name: torch.zeros(
                            x[source].size(0), self.hidden_features
                        ).to(x[source].device)
                    }
                )
        return x

    def forward(self, x):
        x = self.initialize_hidden(x)
        outputs = []
        for _ in range(self.num_iter):
            x = self.combine(x)
            x = self.layer(x)
            outputs.append(self.head(x))

        return outputs
