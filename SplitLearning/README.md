# my-first-github-project

## Split Learning Code Sample

This repository contains a self-contained Python implementation of **split learning**, a privacy-preserving distributed machine learning technique.

### What is Split Learning?

In split learning the neural network is divided into two parts at a *cut layer*:

| Side   | Responsibility                                                                                                                           |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Client | Holds the raw data and runs the first few layers of the network. Sends only the intermediate activations ("smashed data") to the server. |
| Server | Receives the smashed data, completes the forward pass, computes the loss, and sends gradients back to the client.                        |

Because the server never receives the raw input, the client's data remains private.

### Files

| File                   | Description                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `split_learning.py`  | Full split-learning implementation with `ClientModel`, `ServerModel`, `SplitLearningTrainer`, and a training loop on synthetic data. |
| splitLearning_separate | separated model with client_model.py, server_model.py, <br />split_learning_trainer.py, dataset.pyand main.py.                             |

### Requirements

```
pip install torch
```

### Running the sample

```bash
python split_learning.py
```

Expected output (values will vary slightly):

```
Split Learning Training
========================================
  Input size       : 20
  Cut-layer size   : 32
  Num classes      : 5
  Train samples    : 800
  Val samples      : 200
========================================
Epoch  1/10 | Train loss: 1.1234 | Val loss: 0.9876 | Val acc: 0.7150
...
Epoch 10/10 | Train loss: 0.0312 | Val loss: 0.0287 | Val acc: 1.0000
========================================
Training complete.
```

### How It Works

The `SplitLearningTrainer.train_step()` method implements the full protocol:

1. **Client forward pass** — raw data → smashed data.
2. **Transfer smashed data** to the server (detached tensor, so no gradient flows yet).
3. **Server forward pass** — smashed data → predictions → loss.
4. **Server backward pass** — gradients are computed w.r.t. the smashed data.
5. **Transfer gradients back** to the client.
6. **Client backward pass** — gradients flow through the client model.
7. **Both optimizers step** — weights on both sides are updated.
