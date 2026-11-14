# Assignment 3

Text-guided segmentation on RefCOCO: given an image and a natural-language expression naming one
object in it, predict the mask of that object alone.

| Deliverable | Link |
| --- | --- |
| Dataset proposal | [`PROPOSAL.md`](PROPOSAL.md) |
| Assignment page | <https://longlephamtien.github.io/DL261-Assignment/assignments/assignment-3/> |
| AI usage | [`AI_USAGE.md`](../../AI_USAGE.md) |

## Workflow

Set up the environment first, see [assignments](../README.md). No account is needed: the RefCOCO
annotations and the COCO images are both public.

```sh
cd assignment-3

# 1. annotations: fetch refcoco.zip, verify its checksum, unpack
python -m src.data --config configs/base.yaml --annotations

# 2. split: materialise the UNC partition as splits/refcoco-unc.json and check it
python -m src.data --config configs/base.yaml --split

# 3. statistics: write results/eda/distribution.json, which every number in the proposal comes from
python -m src.data --config configs/base.yaml --stats

# 4. images: fetch only the 19,994 frames the expressions refer to
python -m src.data --config configs/base.yaml

# 5. train: one model config per invocation
python -m src.train --config configs/image-only.yaml
python -m src.train --config configs/simple-fusion.yaml --set seed=1
```

Step 1 refuses an archive whose SHA-256 does not match the one committed in `configs/base.yaml`.
The original host no longer resolves, so the configured URL is the Internet Archive copy of that
exact path; checking it is what makes an archived mirror usable as a source.

Step 2 commits the partition so a reported number is traceable without re-downloading, and
refuses a split whose image lists overlap.

Step 4 fetches images in parallel rather than the 13.5 GB COCO training archive, because RefCOCO
reads 19,994 of its 82,783 frames. Measured here: 59 images per second against 4.3 MB/s for the
archive.

## Layout

```
PROPOSAL.md     the dataset proposal, section 27
configs/        base.yaml plus one file per model
splits/         the committed UNC partition
notebooks/      eda.ipynb, compute.ipynb
results/eda/    distribution.json, findings.json, and the figures
tests/          contract tests
src/
  interfaces.py   the paired batch, the encoder, fusion and evaluation contracts
  data.py         download, checksum, split, pair statistics, dataset and loaders
  metrics.py      per-pair IoU and Dice
  models/         registry; one module per modality and the fusions
  train.py        training entry point, shared by every model
  evaluate.py     metrics, latency and prediction overlays
  figures.py      curves and the comparison table
  publish.py      upload a run to the Hub
  utils.py        configuration, paths, run directories
```

## Interfaces

Write against `src/interfaces.py`, not against another module's implementation, so the encoder,
fusion and evaluation work can proceed in parallel. Every stub raises `NotImplementedError`
naming the issue that fills it.

Every model takes `(images, tokens, attention)` and returns per-pixel logits `(N, 2, H, W)`. A
unimodal baseline ignores the argument it does not read, so one training pass serves all four and
the two baselines cannot drift apart from the fusion models through a change applied to only one.

Tests cover the split guarantee, the subset rule, the vocabulary source and the batch contract:

```sh
python -m pytest tests/ -q
```

`notebooks/compute.ipynb` measures the per-step and per-pair cost of all four models and writes
`results/compute.json`. It selects CUDA, then MPS, then CPU, and picks the attention kernel to
match, since the Apple backend has none that supports dropout.

## Artifacts

| Artifact | Location | In git |
| --- | --- | --- |
| RefCOCO annotations | `data/refcoco/` | no, re-downloadable and checksummed |
| COCO images | `data/images/` | no, licensed per image |
| Committed split | `splits/refcoco-unc.json` | yes |
| Run directory | `results/runs/<run_id>/` | no |
| Measured statistics | `results/eda/distribution.json` | yes |
| EDA findings | `results/eda/findings.json` | yes |
| Measured cost | `results/compute.json` | yes |
