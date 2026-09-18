# Assignment 1

A controlled comparison of linear, convolutional, recurrent, and Transformer classifiers on Fashion-MNIST, under one shared split and evaluation protocol. MNIST is for debugging only.

## Workflow

Set up the environment and credentials first, see [assignments](../README.md). Then, from this folder:

```sh
# 0. chore: change directory to this assignment path
cd assignment-1

# 1. data: download Fashion-MNIST and write the shared split
python -m src.data --config configs/base.yaml

# 2. train: one run directory per invocation
python -m src.train --config configs/linear.yaml
python -m src.train --config configs/mlp.yaml --set seed=1

# 3. publish the selected run to the Hugging Face Hub
python -m src.publish results/runs/<run_id>
```

Step 1 runs once per split; steps 2 and 3 run per model and seed.

## Layout

```
configs/      base.yaml plus one file per model
notebooks/    exploratory data analysis
splits/       committed split indices
results/      runs/ (ignored), figures/, eda/
src/
  interfaces.py   shared types, constants, and schemas
  data.py         split, Dataset, DataLoader, sequence adapters
  models/         registry; one module per architecture
  train.py        training entry point
  publish.py      upload a run to the Hub
  utils.py        configuration, seeding, run tracking
```

## Interfaces

Write against `src/interfaces.py`, not against another module's implementation, so the data, training, model, and evaluation work can proceed in parallel. Every stub raises `NotImplementedError` naming the issue that fills it.

**Batch.** `(images, labels)`, images `(N, 1, 28, 28)` float32, labels `(N,)` int64. Normalization statistics come from the training split only. Sequence models take the same batch and call `data.to_sequence`.

**Model registry.** Models are registered by name and built from configuration, so the trainer never imports a model module.

```python
from src.models import register, build_model

@register("linear")
def build_linear(**kwargs) -> nn.Module: ...

model = build_model(config["model"]["name"], **config["model"]["args"])
```

Every model returns **raw logits** `(N, 10)`. Never apply softmax; `CrossEntropyLoss` does it.

**Sequence adapters.** `data.to_sequence(images, representation, patch_size)` returns `(N, T, F)`.

| representation | shape |
| --- | --- |
| `rows`, `columns` | `(N, 28, 28)` |
| `patches` | `(N, (28 / patch_size) ** 2, patch_size ** 2)` |

**Run directory.** One per run, under `results/runs/<run_id>/`.

```
config.yaml        resolved configuration
environment.json   commit hash, library versions, hardware
history.json       one EpochRecord per epoch
summary.json       RunSummary, written once
checkpoint.pt      selected by train.checkpoint_metric
```

## Artifacts

| Artifact | Location | In git |
| --- | --- | --- |
| Fashion-MNIST download | `data/` | no, public and re-downloadable |
| Split indices | `splits/<dataset>-seed<seed>.json` | yes |
| Run directory | `results/runs/<run_id>/` | no |
| Selected checkpoints | Hugging Face, see [shared](../shared/README.md) | no |
| Figures and metric tables | `results/figures/`, `results/eda/` | yes |

The split is committed rather than regenerated, because every model must use exactly the same partition and regenerating from a seed would tie it to specific NumPy and PyTorch versions.

`seed` lives in `configs/base.yaml`, is overridden with `--set seed=1`, and is recorded in each `summary.json`. `utils.set_seed` covers Python, NumPy, and PyTorch, including CUDA and cuDNN determinism.

## Protocol

Fixed across all five models: one split file, one seed list, identical preprocessing and augmentation, and the same checkpoint rule, by default best `val_macro_f1`. The test split is evaluated once, after all model selection is finished.

## Status

Scaffold only. Stubs name their issue: #12 configuration and run tracking, #13 split, #14 Dataset and DataLoaders, #16 training loop, #17 linear, #18 MLP, #39 Hub upload.
