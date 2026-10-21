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
python -m src.train --config configs/cnn.yaml
python -m src.train --config configs/transformer.yaml
python -m src.train --config configs/recurrent.yaml

# repeat each model across the seed list to measure run-to-run variance
python -m src.train --config configs/mlp.yaml --set seed=1
python -m src.train --config configs/mlp.yaml --set seed=2

# override any key for a single run, dotted and repeatable
python -m src.train --config configs/mlp.yaml --set train.lr=1e-3 --set model.args.dropout=0.5

# 3. evaluate: metrics, latency, and prediction examples for one run
python -m src.evaluate <run_id>
python -m src.evaluate <run_id> --split test                # once the run is selected

# 4. figures: compare evaluated runs; each run already holds its own curves
python -m src.figures <run_id> <other_run_id>
python -m src.figures results/runs/*-seed0-* --name architectures
python -m src.figures <run_id> --split test                 # after model selection

# 5. publish: upload the run with every report and example it has
python -m src.publish <run_id> --dry-run                    # list the files, upload nothing
python -m src.publish <run_id> --message "Selected MLP baseline for the M1 draft"
```

Tests cover the model, evaluation, and comparison contracts, and run in a few seconds:

```sh
python -m pytest tests/ -q
```

Step 1 runs once per split; steps 2, 3 and 5 run per model and seed; step 4 runs once per comparison. Steps 4 and 5 read the reports written by step 3, so a run is always evaluated before it is compared or published.

## Layout

```
EXPERIMENTS.md  the experiment plan and fairness protocol
configs/      base.yaml plus one file per model
notebooks/    exploratory data analysis
splits/       committed split indices
tests/        tests for the models, the evaluation, and the comparison
results/      figures/, eda/; runs/ and evaluation/ are ignored
src/
  interfaces.py   shared types, constants, and schemas
  data.py         split, Dataset, DataLoader, sequence adapters
  models/         registry; one module per architecture
  train.py        training entry point
  evaluate.py     metrics, latency, and prediction examples
  figures.py      learning curves and the comparison table
  publish.py      upload a run to the Hub
  utils.py        configuration, seeding, run tracking, result paths
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

A sequence model receives the same image batch as every other model and calls `to_sequence` itself, so the trainer stays free of model-specific branches. The representation and patch size live in the model's own config under `model.args`, and `configs/<model>.yaml` repeats them under `input` so each run records which representation produced it.

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
| Reports and prediction examples | `results/evaluation/<run_id>/<split>/` | no |
| Published runs, reports and examples | Hugging Face, see [shared](../shared/README.md) | no |
| Figures and metric tables | `results/figures/<comparison>/`, `results/eda/` | yes |

The split is committed rather than regenerated, because every model must use exactly the same partition and regenerating from a seed would tie it to specific NumPy and PyTorch versions.

`seed` lives in `configs/base.yaml`, is overridden with `--set seed=1`, and is recorded in each `summary.json`. `utils.set_seed` covers Python, NumPy, and PyTorch, including CUDA and cuDNN determinism.

## Protocol

All five models share one split file, one seed list, identical preprocessing and augmentation, and the checkpoint rule `val_macro_f1`. The test split is evaluated once, after model selection concludes.

[EXPERIMENTS.md](EXPERIMENTS.md) shows the fixed protocol, the seed count and tuning budget, what each experiment varies, and the rule that decides when one model counts as better than another.

## Traceability

Every figure in the report and on the assignment page comes from the runs below. A run id resolves to `results/runs/<run_id>/`, which holds the resolved `config.yaml`, the `environment.json` recording the commit and library versions, the per-epoch `history.json`, and `checkpoint.pt`. Evaluation reports live beside it under `results/evaluation/<run_id>/<split>/`.

Configurations evaluated on `val+test` are the five that form the main architecture comparison. The remaining five are ablation-only and are judged on validation alone, so the test split is read once per configuration and never participates in a selection decision.

| Model | Representation | Config | Splits | Seeds | Run IDs | Commit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Linear | image | `configs/linear.yaml` | val+test | 0, 1, 2 | `linear-seed0-20260922-215714`<br>`linear-seed1-20261019-222817`<br>`linear-seed2-20261019-225007` | `3af46a42`, `42f775db` |
| MLP | image | `configs/mlp.yaml` | val+test | 0, 1, 2 | `mlp-seed0-20260922-220156`<br>`mlp-seed1-20261019-223317`<br>`mlp-seed2-20261019-225325` | `3af46a42`, `42f775db` |
| CNN | image | `configs/cnn.yaml` | val+test | 0, 1, 2 | `cnn-seed0-20261019-231441`<br>`cnn-seed1-20261019-232916`<br>`cnn-seed2-20261019-234256` | `42f775db` |
| GRU | rows | `configs/recurrent.yaml` | val+test | 0, 1, 2 | `recurrent-seed0-20261019-232133`<br>`recurrent-seed1-20261019-233538`<br>`recurrent-seed2-20261019-234927` | `42f775db` |
| GRU | columns | `configs/recurrent.yaml` | val | 0, 1, 2 | `recurrent-seed0-20261020-120428`<br>`recurrent-seed1-20261020-123901`<br>`recurrent-seed2-20261020-204209` | `42f775db` |
| GRU | patches | `configs/recurrent.yaml` | val | 0, 1, 2 | `recurrent-seed0-20261020-121141`<br>`recurrent-seed1-20261020-124637`<br>`recurrent-seed2-20261020-203312` | `42f775db` |
| LSTM | rows | `configs/recurrent.yaml` (`model.args.cell_type=lstm`) | val | 0, 1, 2 | `recurrent-seed0-20261020-210753`<br>`recurrent-seed1-20261020-211421`<br>`recurrent-seed2-20261020-212048` | `42f775db` |
| Transformer | rows | `configs/transformer.yaml` | val | 0, 1, 2 | `transformer-seed0-20261020-122036`<br>`transformer-seed1-20261020-125532`<br>`transformer-seed2-20261020-204923` | `42f775db` |
| Transformer | columns | `configs/transformer.yaml` | val | 0, 1, 2 | `transformer-seed0-20261020-122954`<br>`transformer-seed1-20261020-130444`<br>`transformer-seed2-20261020-205839` | `42f775db` |
| Transformer | patches | `configs/transformer.yaml` | val+test | 0, 1, 2 | `transformer-seed0-20260928-195935`<br>`transformer-seed1-20261019-223849`<br>`transformer-seed2-20261019-225919` | `42f775db`, `aa518436` |

Reproduce any row with the commands in [Workflow](#workflow): train with the listed config at the listed seed, then evaluate the resulting run id on the listed splits.
