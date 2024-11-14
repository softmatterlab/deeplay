from typing import Optional

import torch.nn as nn
import torch
from deeplay.module import DeeplayModule

class GlobalGraphPooling(DeeplayModule):
    """
    Pools all the nodes of the graph to a single cluster.

    (Inspired by MinCut-pooling ('Spectral Clustering with Graph Neural Networks for Graph Pooling'):
    but with the assignment matrix S being deterministic (all nodes are pooled into one cluster))

    Input
    -----
    X: float (Any, Any)  #(number of nodes, number of features)
    
    Output
    ------
    X: float (1, Any)    #(number of clusters, number of features)
    S: float (Any, 1)    #(number of nodes, number of clusters)    
    """

    def __init__(
            self,
            ):
        super().__init__()

        class Select(DeeplayModule):
            def forward(self, x):
                return torch.ones((x.shape[0], 1))  
            
        class ClusterMatrixForBatch(DeeplayModule):
            def forward(self, S, B):
                unique_graphs = torch.unique(B)
                num_graphs = len(unique_graphs)

                S_ = torch.zeros(S.shape[0] * num_graphs)

                row_indices = torch.arange(S.shape[0])
                col_indices = B

                S_[row_indices * num_graphs + col_indices] = S.view(-1)
                B_ = torch.arange(num_graphs)

                return S_.reshape([S.shape[0], -1]), B_
            

        class Reduce(DeeplayModule):
            def forward(self, x, s):
                return torch.matmul(s.transpose(-2,-1), x)

        self.select = Select()
        self.select.set_input_map('x')
        self.select.set_output_map('s')

        self.batch_compatible = ClusterMatrixForBatch()
        self.batch_compatible.set_input_map('s', 'batch')
        self.batch_compatible.set_output_map('s', 'batch')

        self.reduce = Reduce()
        self.reduce.set_input_map('x', 's')
        self.reduce.set_output_map('x')

    def forward(self, x):
        x = self.select(x)
        x = self.batch_compatible(x)
        x = self.reduce(x)
        return (x)

    
class GlobalGraphUpsampling(DeeplayModule):
    """
    Reverse of GlobalGraphPooling.
    Only upsampling the node features.
    """

    def __init__(
            self,
            ):
        super().__init__()
    
        class Upsample(DeeplayModule):
            def forward(self, x, s):
                return torch.matmul(s, x)
            
        self.upsample = Upsample()
        self.upsample.set_input_map('x', 's')
        self.upsample.set_output_map('x')

    def forward(self, x):
        x = self.upsample(x)
        return x