"""
Entry point for the split learning demo.
"""

import torch
from torch.utils.data import DataLoader

from client_model import ClientModel
from dataset import make_dataset
from server_model import ServerModel
from split_learning_trainer import SplitLearningTrainer


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
