from typing import Dict, Any, Union, Tuple, Dict

from deeplay import DeeplayModule

import torch
from torch_geometric.data import Data


class FromDict(DeeplayModule):
    def __init__(self, *keys: str):
        super().__init__()
        self.keys = keys

    def forward(self, x: Dict[str, Any]) -> Union[Any, Tuple[Any, ...]]:
        return (
            x[self.keys[0]]
            if len(self.keys) == 1
            else tuple(x[key] for key in self.keys)
        )

    def extra_repr(self) -> str:
        return ", ".join(self.keys)


class AddDict(DeeplayModule):
    """
    Element-wise addition of two dictionaries.

    Parameters
    ----------
    keys : Tuple[str]
        Specifies the keys to be added element-wise.

    Constraints
    -----------
    - Both dictionaries 'x' (base) and 'y' (addition) must contain the same keys for the addition operation.

    - 'x': Dict[str, Any] or torch_geometric.data.Data.
    - 'y': Dict[str, Any] or torch_geometric.data.Data.
    """

    def __init__(self, *keys: str):
        super().__init__()
        self.keys = keys

    def forward(
        self, x: Union[Dict[str, Any], Data], y: Dict[str, Any]
    ) -> Union[Dict[str, Any], Data]:

        if isinstance(x, Data):
            x = x.clone()
        else:
            x = x.copy()

        x.update({key: torch.add(x[key], y[key]) for key in self.keys})
        return x


class CatDictElements(DeeplayModule):
    """
    Concatenates specified elements within a dictionary-like structure along a given dimension.

    Parameters:
    - keys: Tuple of tuples, where each tuple contains the source and target keys to concatenate.
    - dim: Dimension along which concatenation occurs. Default is -1.
    """

    def __init__(self, *keys: Tuple[tuple], dim: int = -1):
        super().__init__()
        self.source, self.target = zip(*keys)
        self.dim = dim

    def forward(self, x: Union[Dict[str, Any], Data]) -> Union[Dict[str, Any], Data]:
        x = x.clone() if isinstance(x, Data) else x.copy()
        x.update(
            {
                t: torch.cat([x[t], x[s]], dim=self.dim)
                for t, s in zip(self.target, self.source)
            }
        )
        return x
