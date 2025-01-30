"""Graph augmentations for MAGIK.

This module provides classes to augment data during training
with transformations, node dropouts, and noise.

Module Structure
----------------

- `NoisyNode`: Adds random noise to each node.

- `NodeDropout`: Randomly removes a small ammount of nodes and edges.

- `RandomRotation`: Randomly rotates all nodes by the same angle.

- `RandomFlip`: Flips nodes with a 0.5 chance.

- `AugmentCentroids`: Random rotation and translation of nodes.


"""

from math import sin, cos

import numpy as np
import torch
from torch_geometric.data import Data

class NoisyNode:
    """Class to add noise to node attributes.

    """

    def __call__(
        self,
        graph: Data,
    ) -> Data :
        
        # Ensure original graph is unchanged.
        graph = graph.clone()
        
        # Center positions.
        node_feats = graph.x[:, :2] - 0.5
        node_feats += np.random.randn(*node_feats.shape) * np.random.rand()*0.1

        # Restore positions.
        graph.x[:, :2] = node_feats + 0.5
        return graph


class NodeDropout:
    """Removal (dropout) of random nodes to simulate missing frames.

    """

    def __call__(
        self,
        graph: Data
    ) -> Data:

        # Ensure original graph is unchanged.
        graph = graph.clone()

        # Specify node dropout rate.
        dropout_rate = 0.05

        # Get indices of random nodes.
        idx = np.array(list(range(len(graph.x))))
        dropped_idx = idx[np.random.rand(len(graph.x)) < dropout_rate]

        # Compute connectivity matrix to dropped nodes.
        for dropped_node in dropped_idx:
            edges_connected_to_removed_node = np.any(
                np.array(graph.edge_index) == dropped_node, axis=0
            )

        # Remove edges, weights, labels connected to dropped nodes with the
        # bitwise not operator '~'.
        graph.edge_index = graph.edge_index[:, ~edges_connected_to_removed_node]
        graph.edge_attr = graph.edge_attr[~edges_connected_to_removed_node]
        graph.distance = graph.distance[~edges_connected_to_removed_node]
        graph.y = graph.y[~edges_connected_to_removed_node]

        return graph


class RandomRotation:
    """Random rotations to diversify training data.
    
    """
    
    def __call__(
        self,
        graph: Data
    ) -> Data:
        # Ensure original graph is unchanged.
        graph = graph.clone()

        # Center positons.
        node_feats = graph.x[:, :2] - 0.5  
        angle = np.random.rand() * 2 * np.pi

        rotation_matrix = torch.tensor(
            [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        ).float()
        rotated_node_attr = torch.matmul(node_feats, rotation_matrix)

        # Restore positons.
        graph.x[:, :2] = rotated_node_attr + 0.5  
        
        return graph

 
class RandomFlip:
    """Random flip to diversify training data.
    
    """

    def __call__(
        self,
        graph: Data
    ) -> Data:

        # Ensure original graph is unchanged.
        graph = graph.clone()

        # Center positons.
        node_feats = graph.x[:, :2] - 0.5  

        if np.random.randint(2): node_feats[:, 0] *= -1
        if np.random.randint(2): node_feats[:, 1] *= -1

        # Restore positons.
        graph.x[:, :2] = node_feats + 0.5  
        return graph


class AugmentCentroids:
    """Translation and rotation to diversify training data.
    
    """

    def __call__(
        self,
        graph: Data
    ) -> Data:

        graph = graph.clone()

        # Center positions.
        centroids = graph.x[:, :2] - 0.5 

        angle = np.random.rand() * 2 * np.pi
        translate = np.random.rand(1,2)

        # Rotate x component of centroids.
        centroids_x = (
            centroids[:, 0] * np.cos(angle) +
            centroids[:, 1] * np.sin(angle) +
            translate[0]
        )

        # Rotate y component of centroids.
        centroids_y = (
            centroids[:, 1] * np.cos(angle) +
            centroids[:, 0] * np.sin(angle) +
            translate[1]
        )

        # Flip centroids randomly.
        flip = np.random.rand(1,2)

        if flip[0] > 0.5:
            centroids_x *= -1

        if flip[1] > 0.5:
            centroids_y *= -1

        # Restore positions.
        graph.x[:, 0] = centroids_x + 0.5
        graph.x[:, 1] = centroids_y + 0.5

        return graph

