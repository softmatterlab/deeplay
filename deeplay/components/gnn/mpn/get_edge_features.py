from .cla import CombineLayerActivation


class GetEdgeFeatures(CombineLayerActivation):
    """"""

    def get_forward_args(self, x):      # maybe use Tranform instead, and just take the first two ouputs
        """Get the arguments for the ... module.
        An MPN ... module takes the following arguments:
        - node features of sender nodes (x[A[0]])
        - node features of receiver nodes (x[A[1]])
        
        A is the adjacency matrix of the graph.
        """
        x, edge_index = x
        return x[edge_index[0]], x[edge_index[1]]
