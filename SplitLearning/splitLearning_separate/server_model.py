"""
Server model definition for split learning.
"""

import torch
import torch.nn as nn


class ServerModel(nn.Module):
    """Back-end network that lives on the server side.

    Receives the intermediate activations from the client, completes the
    forward pass, and returns the final predictions.
    """

    def __init__(self, cut_layer_size: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cut_layer_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, smashed_data: torch.Tensor) -> torch.Tensor:
        return self.layers(smashed_data)
