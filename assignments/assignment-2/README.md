# Assignment 2

Semantic segmentation on Cityscapes: a U-Net written from scratch against a pretrained DeepLabV3, under one shared split and evaluation protocol.

## Workflow

Set up the environment first, see [assignments](../README.md). The Cityscapes download is account-gated, so add your [registered account](https://www.cityscapes-dataset.com/register/) to the repository `.env` alongside the Hugging Face keys:

```
CITYSCAPES_USERNAME=
CITYSCAPES_PASSWORD=
```

Then, from this folder:

```sh
# 0. chore: change directory to this assignment path
cd assignment-2

# 1. data: download the official archives, unpack them, write the city-level split
python -m src.data --config configs/base.yaml
python -m src.data --download-only                          # fetch the archives and stop

# 2. train: one model config per invocation
python -m src.train --config configs/unet.yaml
python -m src.train --config configs/deeplabv3.yaml

# repeat each model across the seed list to measure run-to-run variance
python -m src.train --config configs/deeplabv3.yaml --set seed=1

# the controlled experiment required by section 20
python -m src.train --config configs/deeplabv3.yaml --set model.args.freeze_backbone=true
python -m src.train --config configs/deeplabv3.yaml --set train.loss=dice

# 3. evaluate: metrics, latency, and prediction overlays for one run
python -m src.evaluate <run_id>

# 4. figures: compare evaluated runs
python -m src.figures <run_id> <other_run_id>

# 5. publish: upload the run with every report and overlay it has
python -m src.publish <run_id> --dry-run
python -m src.publish <run_id> --message "Selected DeepLabV3 baseline for the M2 draft"
```

Tests cover the scaffold contract and run in a second:

```sh
python -m pytest tests/ -q
```

Steps 3 to 5 take a run id as well as a path to the run directory.

## Layout

```
configs/      base.yaml plus one file per model
notebooks/    exploratory data analysis
splits/       committed split indices
tests/        contract tests
results/      figures/, eda/; runs/ and evaluation/ are ignored
src/
  interfaces.py   shared types, constants, and schemas
  data.py         download, city split, mask remapping, DataLoader
  metrics.py      confusion matrix, mIoU, Dice
  models/         registry; one module per architecture
  train.py        training entry point
  evaluate.py     metrics, latency, and prediction overlays
  figures.py      learning curves and the comparison table
  publish.py      upload a run to the Hub
  utils.py        configuration, seeding, run tracking, result paths
```

## Interfaces

Write against `src/interfaces.py`, not against another module's implementation, so the data, training, model, and evaluation work can proceed in parallel. Every stub raises `NotImplementedError` naming the issue that fills it.

**Batch.** `(images, masks)`, images `(N, 3, H, W)` float32, masks `(N, H, W)` int64. Mask values are trainIds in `[0, 19)`, or `IGNORE_INDEX` for pixels carrying no evaluated label. Normalization statistics come from the training split only.

**Model registry.** Models are registered by name and built from configuration, so the trainer never imports a model module.

```python
from src.models import register, build_model

@register("unet")
def build_unet(**kwargs) -> nn.Module: ...

model = build_model(config["model"]["name"], **config["model"]["args"])
```

Every model returns **per-pixel logits** `(N, 19, H, W)` at the input resolution. Never apply softmax; `CrossEntropyLoss` does it, and it takes `ignore_index=255`.

**Classes.** Cityscapes ships 34 labels but is scored on 19. `csCreateTrainIdLabelImgs` from the official toolkit converts `labelIds` to `trainIds` offline, which is cheaper than remapping on every epoch.

**Run directory.** One per run, under `results/runs/<run_id>/`.

```
config.yaml        resolved configuration
environment.json   commit hash, library versions, hardware, thread count
history.json       one EpochRecord per epoch
summary.json       RunSummary, written once
checkpoint.pt      selected by train.checkpoint_metric
curves.png         loss and mIoU
```

## Artifacts

| Artifact | Location | In git |
| --- | --- | --- |
| Cityscapes archives | `data/` | no, account-gated and re-downloadable |
| Split file | `splits/<dataset>-seed<seed>.json` | yes |
| Run directory | `results/runs/<run_id>/` | no |
| Reports and overlays | `results/evaluation/<run_id>/<split>/` | no |
| Published runs, reports and overlays | Hugging Face, see [shared](../shared/README.md) | no |
| Figures and metric tables | `results/figures/<comparison>/`, `results/eda/` | yes |

The split is committed rather than regenerated, because every model must use exactly the same partition.

## Protocol

The split unit is the **city**: train and validation hold disjoint sets of cities, so no street scene reaches both. Cityscapes withholds the test labels, so the official validation split is what gets reported, and model selection uses a held-out slice of train.

All models share one split file, one seed list, identical preprocessing and augmentation, and the checkpoint rule `val_miou`.
