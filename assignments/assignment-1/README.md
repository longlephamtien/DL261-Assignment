# Assignment 1

A controlled comparison of linear, convolutional, recurrent, and Transformer classifiers on Fashion-MNIST, under one shared split and evaluation protocol. MNIST is for debugging only.

## Workflow

Set up the environment and credentials first, see [assignments](../README.md). Then, from this folder:

```sh
# 0. chore: change directory to this assignment path
cd assignment-1

# 1. data: download Fashion-MNIST, write the shared split, export an example batch
python -m src.data --config configs/base.yaml

# 2. train: one model config per invocation
python -m src.train --config configs/linear.yaml
python -m src.train --config configs/mlp.yaml

# repeat each model across the seed list to measure run-to-run variance
python -m src.train --config configs/mlp.yaml --set seed=1
python -m src.train --config configs/mlp.yaml --set seed=2

# override any key for a single run, dotted and repeatable
python -m src.train --config configs/mlp.yaml --set train.lr=1e-3 --set model.args.dropout=0.5

# 3. figures: compare finished runs; each run already holds its own curves
python -m src.figures results/runs/<linear_run> results/runs/<mlp_run>

# 4. publish the selected run to the Hugging Face Hub
python -m src.publish results/runs/<run_id>
```

Tests cover the model contract and run in a second:

```sh
python -m pytest tests/ -q
```

Step 1 runs once per split; steps 2 and 4 run per model and seed, step 3 runs once per comparison.

## Layout

```
configs/      base.yaml plus one file per model
notebooks/    exploratory data analysis
splits/       committed split indices
tests/        model contract tests
results/      runs/ (ignored), figures/, eda/
src/
  interfaces.py   shared types, constants, and schemas
  data.py         split, Dataset, DataLoader, sequence adapters
  models/         registry; one module per architecture
  train.py        training entry point
  figures.py      learning curves and the comparison table
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
environment.json   commit hash, library versions, hardware, thread count
history.json       one EpochRecord per epoch
summary.json       RunSummary, written once
checkpoint.pt      selected by train.checkpoint_metric
curves.png         loss and macro-F1
```

## Artifacts

| Artifact | Location | In git |
| --- | --- | --- |
| Fashion-MNIST download | `data/` | no, public and re-downloadable |
| Split indices | `splits/<dataset>-seed<seed>.json` | yes |
| Run directory | `results/runs/<run_id>/` | no |
| Selected checkpoints | Hugging Face, see [shared](../shared/README.md) | no |
| Evaluation reports | `results/evaluation/<run_id>.json` | yes |
| Correct and incorrect predictions | `results/examples/<run_id>/` | yes |
| Figures and metric tables | `results/figures/`, `results/eda/` | yes |

The split is committed rather than regenerated, because every model must use exactly the same partition and regenerating from a seed would tie it to specific NumPy and PyTorch versions.

`seed` lives in `configs/base.yaml`, is overridden with `--set seed=1`, and is recorded in each `summary.json`. `utils.set_seed` covers Python, NumPy, and PyTorch, including CUDA and cuDNN determinism.

## Protocol

Fixed across all five models: one split file, one seed list, identical preprocessing and augmentation, and the same checkpoint rule, by default best `val_macro_f1`. The test split is evaluated once, after all model selection is finished.
