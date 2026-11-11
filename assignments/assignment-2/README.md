# Assignment 2

Semantic segmentation on Cityscapes: a U-Net written from scratch against a pretrained DeepLabV3, under one shared split and evaluation protocol.

| Deliverable | Link |
| --- | --- |
| Assignment page | <https://longlephamtien.github.io/DL261-Assignment/assignments/assignment-2/> |
| Dataset proposal | [`PROPOSAL.md`](PROPOSAL.md) |
| Experiment plan | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| Report | [`report/report.pdf`](report/report.pdf) |
| Slides | [`presentation/presentation.pdf`](presentation/presentation.pdf) |
| Presentation video | <https://youtu.be/BBk91a9ARaM> |
| Checkpoints | <https://huggingface.co/longlephamtien/DL261-Assignment/tree/main/assignment-2/runs> |
| AI usage | [`AI_USAGE.md`](../../AI_USAGE.md) |

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

## Reproducibility

Every reported number traces to a run directory that records the configuration it resolved to, the
software versions it ran under, and the commit it was produced at.

**Environment.** Python 3.12.0, PyTorch 2.14.0, torchvision 0.29.0, NumPy 2.5.3, scikit-learn
1.9.1, installed by `pip install -e .` from [`assignments/pyproject.toml`](../pyproject.toml),
which pins them. Hardware: Apple M4 Pro, 14 cores, 48 GB unified memory, MPS backend, 10 torch
threads. All runs are full `float32`; mixed precision is not used.

**Split.** One committed file, [`splits/cityscapes-seed0.json`](splits/cityscapes-seed0.json),
read by every run and never regenerated: 15 training cities, 3 validation cities, and the 3 cities
of the official Cityscapes validation partition as the test set.

**Seeds.** 0, 1 and 2 for every configuration entering a comparison; the backbone extension runs
at seed 0 only. The seed sets the Python, NumPy and PyTorch generators together and controls
initialisation, crop positions and batch order. The split is the seed 0 file in every case.

**Reproducing the matrix from a clean clone.** After the setup in
[assignments](../README.md) and the Cityscapes credentials above:

```sh
cd assignments/assignment-2
python -m src.data --config configs/base.yaml     # download, unpack, write the split
bash scripts/experiments.sh                       # the thirteen runs, resumable
bash scripts/evaluate_all.sh                      # score every run on val and test
python -m src.figures results/runs/*/ --name all --split val    # the decision rule is applied here
python -m src.figures results/runs/*/ --name all --split test   # the reported figures and tables
```

`src.figures` writes the learning curves, the per-class chart and one confusion matrix per
configuration, and prints the metrics table, the pairwise verdicts under the decision rule, the
per-class comparison and the error taxonomy. Everything is averaged over a configuration's seeds,
figures included. `python -m src.figures --pipeline` writes the pipeline diagram, which depends on
no run.

`scripts/experiments.sh` skips arms already marked done, so it can be interrupted and restarted.
The thirteen runs took 25.5 hours on the hardware above.

One limit is imposed by the hardware. The MPS backend provides no deterministic kernel for the
scatter-add DeepLabV3 reaches in the backward pass of its final interpolation, so PyTorch warns
and proceeds nondeterministically: the U-Net reproduces bit-exactly, while DeepLabV3 runs diverge
by about 4e-07 per step. Rerunning a DeepLabV3 seed will not reproduce its number exactly, and the
seed-to-seed spread reported in the report includes this.

### Pretrained weights

Both pretrained arms start from torchvision's published DeepLabV3 weights, fetched by
`torchvision==0.29.0` at the enum default. Only the final 1x1 convolution is replaced, for the 19
Cityscapes classes; the backbone and the pooling module transfer unchanged. The U-Net uses no
pretrained weights at all.

| Backbone | Weight enum | File |
| --- | --- | --- |
| MobileNetV3-Large | `DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1` | [`deeplabv3_mobilenet_v3_large-fc3c493d.pth`](https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth) |
| ResNet50 | `DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1` | [`deeplabv3_resnet50_coco-cd0a2569.pth`](https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth) |

Both were trained on a COCO subset carrying the twenty VOC classes, over an ImageNet
classification backbone, and both predict 21 classes before the head is replaced. The
normalization the report applies is read from these weights rather than assumed, because feeding
a differently normalised batch to a pretrained backbone degrades it silently.

### Checkpoints

Publishing and reconstruction are the shared mechanism, documented in
[shared](../shared/README.md). All thirteen runs go up in one invocation:

```sh
python -m src.publish results/runs/*/ --dry-run   # check first, upload nothing
python -m src.publish results/runs/*/
```

Each run becomes one Hub commit carrying `checkpoint.pt`, the resolved `config.yaml`,
`environment.json`, `history.json`, `summary.json`, both evaluation reports, and a
`checksums.json` holding the SHA-256 of every file beside it.

### Configurations

| Configuration | Built from |
| --- | --- |
| DeepLabV3-MobileNetV3 | `configs/deeplabv3.yaml` |
| DeepLabV3-MobileNetV3, frozen | `configs/deeplabv3.yaml` + `model.args.freeze_backbone=true` |
| DeepLabV3-MobileNetV3, Dice | `configs/deeplabv3.yaml` + `train.loss=dice` |
| DeepLabV3-ResNet50 | `configs/deeplabv3.yaml` + `model.args.backbone=resnet50` |
| U-Net | `configs/unet.yaml` |

### Traceability

| Configuration | Seed | Run ID | Best epoch | Val mIoU | Test mIoU | Commit |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| DeepLabV3-MobileNetV3 | 0 | `deeplabv3-seed0-20261109-230158` | 30 | 0.6184 | 0.6125 | `567bab03` |
| DeepLabV3-MobileNetV3 | 1 | `deeplabv3-seed1-20261110-003107` | 31 | 0.6143 | 0.6194 | `277d6794` |
| DeepLabV3-MobileNetV3 | 2 | `deeplabv3-seed2-20261110-020225` | 35 | 0.6083 | 0.6220 | `277d6794` |
| DeepLabV3-MobileNetV3, frozen | 0 | `deeplabv3-seed0-20261110-121645` | 15 | 0.4906 | 0.4925 | `277d6794` |
| DeepLabV3-MobileNetV3, frozen | 1 | `deeplabv3-seed1-20261110-125324` | 24 | 0.4998 | 0.4986 | `277d6794` |
| DeepLabV3-MobileNetV3, frozen | 2 | `deeplabv3-seed2-20261110-134416` | 36 | 0.4980 | 0.5098 | `277d6794` |
| DeepLabV3-MobileNetV3, Dice | 0 | `deeplabv3-seed0-20261110-144803` | 4 | 0.4911 | 0.4843 | `277d6794` |
| DeepLabV3-MobileNetV3, Dice | 1 | `deeplabv3-seed1-20261110-151818` | 4 | 0.4963 | 0.5028 | `277d6794` |
| DeepLabV3-MobileNetV3, Dice | 2 | `deeplabv3-seed2-20261110-154830` | 4 | 0.5000 | 0.5076 | `277d6794` |
| DeepLabV3-ResNet50 | 0 | `deeplabv3-seed0-20261110-161840` | 37 | 0.6665 | 0.7076 | `277d6794` |
| U-Net | 0 | `unet-seed0-20261110-033604` | 38 | 0.4687 | 0.4649 | `277d6794` |
| U-Net | 1 | `unet-seed1-20261110-062947` | 40 | 0.4824 | 0.4674 | `277d6794` |
| U-Net | 2 | `unet-seed2-20261110-092324` | 39 | 0.4995 | 0.4782 | `277d6794` |

The first run was launched one commit earlier than the rest; the two commits differ only in
`scripts/experiments.sh`, and in no training, data or evaluation code.

## Layout

```
EXPERIMENTS.md  the experiment plan and decision rule
configs/        base.yaml plus one file per model
notebooks/      exploratory data analysis
splits/         committed split indices
tests/          contract tests
results/        figures/, eda/; runs/ and evaluation/ are ignored
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

All models share one split file, one seed list, identical preprocessing and augmentation, and the checkpoint rule `val_miou`. The test split is evaluated once, after model selection concludes.

[EXPERIMENTS.md](EXPERIMENTS.md) shows the fixed protocol, the fine-tuning procedure, the seed count and budget, what each experiment varies, and the rule that decides when one configuration counts as better than another.
