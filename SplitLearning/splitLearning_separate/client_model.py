"""
Client model definition for split learning.
"""

import torch
import torch.nn as nn


class ClientModel(nn.Module):
    """Front-end network that lives on the client side.

    Takes raw input features and produces intermediate activations that are
    sent to the server. The raw data never leaves the client.
    """

    def __init__(self, input_size: int, cut_layer_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, cut_layer_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
