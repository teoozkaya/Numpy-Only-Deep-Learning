# NumPy Neural Network

<!-- one or two sentences: what this is and why you built it -->

## Status

<!-- keep a checklist of what is implemented -->

- [ ] `core/layers/` — affine, relu, sigmoid, tanh, dropout, batchnorm, conv, pool
- [ ] `core/losses/` — mse, bce, softmax_cross_entropy
- [ ] `core/optim/` — sgd, momentum, rmsprop, adam
- [ ] `core/model.py` — sequential container
- [x] `data/` — MNIST loading, normalization, batching
- [x] `viz/` — loss curves, weight histograms, decision boundaries
- [x] `tests/` — gradient checking, shape and contract tests

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
pytest tests/                      # full suite
pytest tests/test_gradients.py -v  # gradient checks only
python scripts/train_mnist.py      # MNIST baseline
```

## Layout

```
core/     layers, losses, optimizers, sequential model   (NumPy only)
data/     MNIST download, preprocessing, mini-batching
viz/      loss curves, weight histograms, decision boundaries
tests/    numerical gradient checking and shape contracts
scripts/  training entry points
```

## Design notes

<!-- the conventions you settled on: cache handling, (loss, dx) tuples,
     params/grads dicts, batch-first shapes -->

## Results

<!-- MNIST numbers, plots, what each experiment changed -->

## What I learned

<!-- see NOTES.md -->
