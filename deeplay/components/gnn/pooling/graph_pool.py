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
    # select_output_map: Optional[str]

    def __init__(
            self,
            # select_output_map: Optional[str] = "s",
            ):
        super().__init__()

        # self.select_output_map = select_output_map

        class Select(DeeplayModule):
            def forward(self, x):
                return torch.ones((x.shape[0], 1))   # is this the right dim even if we use batches?
            
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
                # return torch.sum(x, dim=0, keepdim=True)
                return torch.matmul(s.transpose(-2,-1), x)

        self.select = Select()
        self.select.set_input_map('x')
        self.select.set_output_map('s') #self.select_output_map)

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
    # select_input_map: Optional[str]

    def __init__(
            self,
            # select_input_map: Optional[str] = "s",
            ):
        super().__init__()
        # self.select_input_map = select_input_map

        class Upsample(DeeplayModule):
            def forward(self, x, s):
                return torch.matmul(s, x)
            
        self.upsample = Upsample()
        # self.upsample.set_input_map('x_pool', 's')
        self.upsample.set_input_map('x', 's')#self.select_input_map)
        self.upsample.set_output_map('x')

    def forward(self, x):
        x = self.upsample(x)
        return x


class MinCutUpsampling(DeeplayModule):
    """
    Reverse of MinCutPooling as described in 'Spectral Clustering with Graph Neural Networks for Graph Pooling'.
    """
    # select_input_map: Optional[str]
    # connect_input_map: Optional[str]

    def __init__(
            self,
            # select_input_map: Optional[str] = "s",
            # connect_input_map: Optional[str] = "edge_index",
            ):
        super().__init__()
        # self.select_input_map = select_input_map
        # self.connect_input_map = connect_input_map

        class Upsample(DeeplayModule):
            def forward(self, x_pool, a_pool, s):
                x = torch.matmul(s, x_pool)

                if a_pool.is_sparse:
                    a = torch.spmm(a_pool, s.T)
                elif (not a_pool.is_sparse) & (a_pool.size(0) == 2):
                    a_pool = torch.sparse_coo_tensor(
                        a_pool,
                        torch.ones(a_pool.size(1)),
                        ((s.T).size(0),) * 2,
                        device=a_pool.device,
                    )
                    a = torch.spmm(a_pool, s.T)
                elif (not a_pool.is_sparse) & len({a_pool.size(0), a_pool.size(1), (s.T).size(0)}) == 1:
                    a = a_pool.type(s.dtype) @ s.T
            
                return x, a
            
        self.upsample = Upsample()
        self.upsample.set_input_map('x', 'edge_index_pool', 's')
        # self.upsample.set_input_map('x', self.connect_input_map, self.select_input_map)
        self.upsample.set_output_map('x', 'edge_index_')

    def forward(self, x):
        x = self.upsample(x)
        return x