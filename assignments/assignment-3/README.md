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
configs/        base.yaml
splits/         the committed UNC partition
notebooks/      exploratory data analysis
results/eda/    distribution.json and its figures
src/
  interfaces.py   shared types and the split constants
  data.py         download, checksum, and the pair statistics
  utils.py        configuration, paths, run directories
```

## Artifacts

| Artifact | Location | In git |
| --- | --- | --- |
| RefCOCO annotations | `data/refcoco/` | no, re-downloadable and checksummed |
| COCO images | `data/images/` | no, licensed per image |
| Committed split | `splits/refcoco-unc.json` | yes |
| Measured statistics | `results/eda/distribution.json` | yes |
