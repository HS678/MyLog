"""
Split Learning Code Sample
==========================
Split learning is a privacy-preserving distributed machine learning technique
where a neural network is split between a client and a server. The client
processes raw data through the initial layers and sends only the intermediate
activations (the "smashed data") to the server. The server processes the
remaining layers, computes the loss, and sends gradients back to the client
for end-to-end backpropagation — without the server ever seeing the raw data.

This sample demonstrates:
  - ClientModel  : the front portion of the network (runs on the client)
  - ServerModel  : the back portion of the network (runs on the server)
  - SplitLearningTrainer : orchestrates the forward/backward pass across the split
  - A simple training loop on synthetic data (no external dependencies beyond PyTorch)

Requirements:
    pip install torch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class ClientModel(nn.Module):
    """Front-end network that lives on the client side.

    Takes raw input features and produces intermediate activations that are
    sent to the server.  The raw data never leaves the client.
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SplitLearningTrainer:
    """Orchestrates the split-learning training protocol.

    Protocol per mini-batch
    -----------------------
    1. **Client forward pass** — client computes smashed data from raw input.
    2. **Send smashed data to server** — in practice this is a network call;
       here it is a simple tensor copy (detach + requires_grad).
    3. **Server forward pass** — server computes predictions and loss.
    4. **Server backward pass** — server computes gradients w.r.t. smashed data.
    5. **Send gradients back to client** — in practice a network call; here a
       simple tensor reference.
    6. **Client backward pass** — client uses the received gradients to finish
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

    # ------------------------------------------------------------------
    # Single-batch training step
    # ------------------------------------------------------------------

    def train_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Execute one split-learning training step and return the scalar loss."""

        # ---- Step 1: Client forward pass ----
        self.client_optimizer.zero_grad()
        smashed_data = self.client_model(inputs)

        # ---- Step 2: Detach smashed data before sending to server ----
        # requires_grad=True so that we can compute gradients w.r.t. it later.
        smashed_data_detached = smashed_data.detach().requires_grad_(True)

        # ---- Step 3: Server forward pass ----
        self.server_optimizer.zero_grad()
        predictions = self.server_model(smashed_data_detached)

        # ---- Step 4: Server backward pass ----
        loss = self.criterion(predictions, labels)
        loss.backward()

        # ---- Step 5: Retrieve gradients for smashed data ----
        smashed_gradients = smashed_data_detached.grad

        # ---- Step 6: Client backward pass ----
        smashed_data.backward(smashed_gradients)

        # ---- Step 7: Update both models ----
        self.server_optimizer.step()
        self.client_optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Synthetic dataset helper
# ---------------------------------------------------------------------------

def make_dataset(
    num_samples: int = 1000,
    input_size: int = 20,
    num_classes: int = 5,
    seed: int = 42,
) -> TensorDataset:
    """Create a simple linearly-separable synthetic classification dataset."""
    torch.manual_seed(seed)
    X = torch.randn(num_samples, input_size)
    # Ground-truth weights that create a separable dataset
    W = torch.randn(input_size, num_classes)
    logits = X @ W
    y = logits.argmax(dim=1)
    return TensorDataset(X, y)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    # Hyper-parameters
    INPUT_SIZE = 20
    CUT_LAYER_SIZE = 32
    NUM_CLASSES = 5
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LR = 1e-3

    # Build dataset / loaders
    dataset = make_dataset(num_samples=1000, input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_inputs = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    val_labels = torch.tensor([val_dataset[i][1] for i in range(len(val_dataset))])

    # Instantiate models and trainer
    client_model = ClientModel(INPUT_SIZE, CUT_LAYER_SIZE)
    server_model = ServerModel(CUT_LAYER_SIZE, NUM_CLASSES)
    trainer = SplitLearningTrainer(client_model, server_model, lr=LR)

    print("Split Learning Training")
    print("=" * 40)
    print(f"  Input size       : {INPUT_SIZE}")
    print(f"  Cut-layer size   : {CUT_LAYER_SIZE}")
    print(f"  Num classes      : {NUM_CLASSES}")
    print(f"  Train samples    : {train_size}")
    print(f"  Val samples      : {val_size}")
    print("=" * 40)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        client_model.train()
        server_model.train()
        epoch_loss = 0.0
        num_batches = 0
        for inputs, labels in train_loader:
            loss = trainer.train_step(inputs, labels)
            epoch_loss += loss
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches

        # Validation
        client_model.eval()
        server_model.eval()
        val_metrics = trainer.evaluate(val_inputs, val_labels)

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f}"
        )

    print("=" * 40)
    print("Training complete.")


if __name__ == "__main__":
    main()
