import torch
from torch import nn
from typing import Optional

from transformers.models.roberta_prelayernorm.modeling_roberta_prelayernorm import (
    RobertaPreLayerNormEncoder,
    RobertaPreLayerNormConfig,
)


class LoopedTransformer(nn.Module):
    """
    A transformer model that loops the RobertaPreLayerNormEncoder n_loop times.
    Takes a graph adjacency matrix as input and predicts a graph adjacency matrix as output.
    """

    def __init__(
        self,
        graph_size: int,
        n_loop: int,
        hidden_size: int,
        read_in_method: str = "linear",  # "linear" or "zero_pad"
        num_attention_heads: int = 1,
    ):
        super().__init__()
        self.config = RobertaPreLayerNormConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=1,
        )
        self.encoder = RobertaPreLayerNormEncoder(self.config)

        self.n_loop = n_loop
        self.hidden_size = hidden_size
        self.graph_size = graph_size  # number of nodes in the graph

        # Read-in layer: Convert adjacency matrix to hidden representation
        if read_in_method == "linear":
            self.read_in = nn.Linear(self.graph_size, hidden_size)
        elif read_in_method == "zero_pad":
            self.read_in = None
        else:
            raise ValueError(f"Unknown read_in_method: {read_in_method}")

        # Read-out layer: Project hidden representation back to adjacency matrix
        self.read_out = nn.Linear(hidden_size, self.graph_size)

    def forward(
        self,
        adjacency_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            adjacency_matrix: Graph adjacency matrix of shape (batch_size, graph_size, graph_size)

        Returns:
            torch.Tensor: Predicted adjacency matrix of shape (batch_size, graph_size, graph_size)
        """
        batch_size = adjacency_matrix.shape[0]
        # Apply read-in layer to get node embeddings
        if self.read_in is not None:
            hidden_states = self.read_in(adjacency_matrix)
        else:
            # Zero padding method: Create embeddings and copy row sums to first dimension
            hidden_states = torch.zeros(
                batch_size,
                self.graph_size,
                self.hidden_size,
                device=adjacency_matrix.device,
                dtype=adjacency_matrix.dtype,
            )
            hidden_states[:, :, self.graph_size] = adjacency_matrix

        # Loop the encoder n_loop times
        layer_outputs = []
        for _ in range(self.n_loop):
            output = self.read_out(hidden_states)
            hidden_states = self.encoder(
                hidden_states=hidden_states,
                attention_mask=None,
            ).last_hidden_state

            layer_outputs.append(output)

        output = self.read_out(hidden_states)
        layer_outputs.append(output)

        return output, layer_outputs
