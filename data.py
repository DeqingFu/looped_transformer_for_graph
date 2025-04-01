import torch
from torch.utils.data import Dataset
import networkx as nx
from torch_geometric.utils import from_networkx, to_dense_adj
from torch_geometric.data import Data
import random
import math
from typing import List, Tuple, Optional
import pdb


class ErdosRenyiGraphDataset(Dataset):
    def __init__(
        self,
        num_samples,
        num_nodes,
        p: Optional[float],
        sample_p: Optional[bool] = False,
        p_range: Optional[Tuple] = (0.0, 1.0),
        add_self_loops: Optional[bool] = True,
    ):
        """
        Initializes the dataset.

        Args:
            num_samples (int): The number of samples (graphs) in the dataset.
            num_nodes (int): The number of nodes in each generated graph.
            p (float): Fixed probability of edge creation in the Erdős-Rényi model if sample_p is False.
            sample_p (bool): If True, sample p from a uniform distribution within p_range.
            p_range (tuple of float): The range (min, max) for sampling p if sample_p is True.
        """
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.p = p
        self.sample_p = sample_p
        self.p_range = p_range
        self.mat_power = self.num_nodes
        self.add_self_loops = add_self_loops

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Determine the probability of edge creation
        if self.sample_p:
            # Sample p from the specified range
            p = random.uniform(self.p_range[0], self.p_range[1])
        else:
            # Use the fixed p
            p = self.p

        # Generate an Erdős-Rényi graph using NetworkX with the chosen p
        G = nx.erdos_renyi_graph(n=self.num_nodes, p=p)

        # Convert NetworkX graph to PyTorch Geometric data format
        data = from_networkx(G)

        # 1. Compute the adjacency matrix
        adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=self.num_nodes)
        if self.add_self_loops:
            adj_matrix += torch.eye(self.num_nodes)
        adj_matrix = adj_matrix.squeeze(0)

        # 2. Compute the connectivity matrix
        # Compute the transitive closure using matrix exponentiation
        identity = torch.eye(self.num_nodes)
        reachability_matrix = (identity + adj_matrix).matrix_power(self.mat_power)
        connectivity_matrix = (reachability_matrix > 0).float()

        return adj_matrix, connectivity_matrix


class ErdosRenyiTwoGraphsDataset(Dataset):
    def __init__(
        self,
        num_samples,
        num_nodes,
        p: Optional[float],
        sample_p: Optional[bool] = False,
        p_range: Optional[Tuple] = (0.0, 1.0),
        add_self_loops: Optional[bool] = True,
    ):
        """
        Initializes the dataset.

        Args:
            num_samples (int): The number of samples (graphs) in the dataset.
            num_nodes (int): The number of nodes in each generated graph.
            p (float): Fixed probability of edge creation in the Erdős-Rényi model if sample_p is False.
            sample_p (bool): If True, sample p from a uniform distribution within p_range.
            p_range (tuple of float): The range (min, max) for sampling p if sample_p is True.
        """
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.p = p
        self.sample_p = sample_p
        self.p_range = p_range
        self.mat_power = self.num_nodes
        self.add_self_loops = add_self_loops

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Determine the probability of edge creation
        if self.sample_p:
            # Sample p from the specified range
            p = random.uniform(self.p_range[0], self.p_range[1])
        else:
            # Use the fixed p
            p = self.p

        # Generate an Erdős-Rényi graph using NetworkX with the chosen p
        while True:
            G1 = nx.erdos_renyi_graph(n=self.num_nodes // 2, p=p)
            largest_comp1 = max(nx.connected_components(G1), key=len)
            if len(largest_comp1) >= self.num_nodes // 4:
                break
        while True:
            G2 = nx.erdos_renyi_graph(n=self.num_nodes // 2, p=p)
            largest_comp2 = max(nx.connected_components(G2), key=len)
            if len(largest_comp2) >= self.num_nodes // 4:
                break

        # Convert NetworkX graph to PyTorch Geometric data format
        G1_data = from_networkx(G1)
        G2_data = from_networkx(G2)

        # 1. Compute the adjacency matrix
        adj_matrix1 = to_dense_adj(
            G1_data.edge_index, max_num_nodes=self.num_nodes // 2
        )
        adj_matrix2 = to_dense_adj(
            G2_data.edge_index, max_num_nodes=self.num_nodes // 2
        )
        adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        adj_matrix[: self.num_nodes // 2, : self.num_nodes // 2] = adj_matrix1
        adj_matrix[self.num_nodes // 2 :, self.num_nodes // 2 :] = adj_matrix2
        if self.add_self_loops:
            adj_matrix += torch.eye(self.num_nodes)
        adj_matrix = adj_matrix.squeeze(0)

        # Find and connect nodes from largest components with 0.5 probability
        if random.random() < 0.5:
            # Get largest components
            largest_comp1 = max(nx.connected_components(G1), key=len)
            largest_comp2 = max(nx.connected_components(G2), key=len)

            # Select random nodes from largest components
            node1 = random.choice(list(largest_comp1))
            node2 = random.choice(list(largest_comp2))

            # Shift node2 index and add edge
            node2_shifted = node2 + self.num_nodes // 2
            adj_matrix[node1, node2_shifted] = 1
            adj_matrix[node2_shifted, node1] = 1

        # 2. Compute the connectivity matrix
        # Compute the transitive closure using matrix exponentiation
        identity = torch.eye(self.num_nodes)
        reachability_matrix = (identity + adj_matrix).matrix_power(self.mat_power)
        connectivity_matrix = (reachability_matrix > 0).float()

        return adj_matrix, connectivity_matrix


if __name__ == "__main__":
    # Example usage:
    # Parameters
    num_samples = 100  # Number of graphs to generate
    num_nodes = 32  # Number of nodes per graph
    fixed_p = 0.1  # Fixed probability of edge creation
    sample_p = True  # Flag to sample p instead of using a fixed value
    p_range = (0.05, 0.2)  # Range for sampling p if sample_p is True

    # Create the dataset
    dataset = ErdosRenyiTwoGraphsDataset(
        num_samples=num_samples,
        num_nodes=num_nodes,
        p=fixed_p,
        sample_p=sample_p,
        p_range=p_range,
    )

    # Access a sample graph and its adjacency matrix
    import time

    start = time.time()
    for i in range(10):
        adj_matrix, connectivity_matrix = dataset[0]
        end = time.time()
        print("Time taken: ", end - start)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("\nConnectivity Matrix:")
        print(connectivity_matrix)
