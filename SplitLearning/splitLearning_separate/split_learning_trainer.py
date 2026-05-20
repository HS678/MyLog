"""
Split learning trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from client_model import ClientModel
from server_model import ServerModel


class SplitLearningTrainer:
    """Orchestrates the split-learning training protocol.

    Protocol per mini-batch
    -----------------------
    1. **Client forward pass** -- client computes smashed data from raw input.
    2. **Send smashed data to server** -- in practice this is a network call;
       here it is a simple tensor copy (detach + requires_grad).
    3. **Server forward pass** -- server computes predictions and loss.
    4. **Server backward pass** -- server computes gradients w.r.t. smashed data.
    5. **Send gradients back to client** -- in practice a network call; here a
       simple tensor reference.
    6. **Client backward pass** -- client uses the received gradients to finish
       backpropagation through its own layers.
    7. **Both optimizers step**.
    """

    def __init__(
        self,
        client_model: ClientModel,
        server_model: ServerModel,
        lr: float = 1e-3,
    ) -> None:
        self.client_model = client_model
        self.server_model = server_model
        self.criterion = nn.CrossEntropyLoss()
        self.client_optimizer = optim.Adam(client_model.parameters(), lr=lr)
        self.server_optimizer = optim.Adam(server_model.parameters(), lr=lr)

    def train_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Execute one split-learning training step and return the scalar loss."""

        # Step 1: Client forward pass
        self.client_optimizer.zero_grad()
        smashed_data = self.client_model(inputs)

        # Step 2: Detach smashed data before sending to server
        # requires_grad=True so that we can compute gradients w.r.t. it later.
        smashed_data_detached = smashed_data.detach().requires_grad_(True)

        # Step 3: Server forward pass
        self.server_optimizer.zero_grad()
        predictions = self.server_model(smashed_data_detached)

        # Step 4: Server backward pass
        loss = self.criterion(predictions, labels)
        loss.backward()

        # Step 5: Retrieve gradients for smashed data
        smashed_gradients = smashed_data_detached.grad

        # Step 6: Client backward pass
        smashed_data.backward(smashed_gradients)

        # Step 7: Update both models
        self.server_optimizer.step()
        self.client_optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> dict:
        """Return loss and accuracy on the given data."""
        smashed_data = self.client_model(inputs)
        predictions = self.server_model(smashed_data)
        loss = self.criterion(predictions, labels).item()
        predicted_classes = predictions.argmax(dim=1)
        accuracy = (predicted_classes == labels).float().mean().item()
        return {"loss": loss, "accuracy": accuracy}
