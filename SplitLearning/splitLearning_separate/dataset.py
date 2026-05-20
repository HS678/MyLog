"""
Synthetic dataset helper for split learning demos.
"""

import torch
from torch.utils.data import TensorDataset


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
