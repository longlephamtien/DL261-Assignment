# Assignment 3 dataset proposal

Text-guided segmentation on RefCOCO. Every number below is measured from the official annotation
release by `python -m src.data --stats`, which writes
[`results/eda/distribution.json`](results/eda/distribution.json).

## Dataset

| Field | Value |
| --- | --- |
| Name | RefCOCO (UNC split), referring expressions over MS COCO 2014 images |
| Source | `refs(unc).p` and `instances.json` from [`lichengunc/refer`](https://github.com/lichengunc/refer); images from [COCO 2014 train](http://images.cocodataset.org/zips/train2014.zip) |
| Version | Release dated 2016-02-04, SHA-256 `7f924bb7ed8dc4568058e4ff626281918d56e5206f4c868c5a80f088f38c8bf0` |
| License | The annotations carry no stated licence of their own; the `refer` toolkit is Apache-2.0. Images are COCO 2014, licensed per image under seven Creative Commons terms declared in `instances.json` |
| Citation | Kazemzadeh et al., ReferItGame, EMNLP 2014; Yu et al., Modeling Context in Referring Expressions, ECCV 2016 |

The original host `bvisionweb1.cs.unc.edu` no longer resolves. The archived copy of that exact URL
serves 53,993,993 bytes, matching the `Content-Length` the original server reported, and is the
copy checksummed above. COCO's image host is live.

**Per-image licensing.** 13,904 of the 19,994 images, 69.5%, carry a NonCommercial term, and
6,353 carry NoDerivs or ShareAlike. Educational use is covered by all seven; redistribution is
not, so the images are downloaded locally and kept out of version control.

| Licence | Images |
| --- | ---: |
| CC BY-NC-SA 2.0 | 5,669 |
| CC BY-NC-ND 2.0 | 5,388 |
| CC BY 2.0 | 3,140 |
| CC BY-NC 2.0 | 2,847 |
| CC BY-SA 2.0 | 1,899 |
| CC BY-ND 2.0 | 965 |
| No known copyright restrictions | 86 |

## Task, modalities, input, output

Text-guided segmentation, one of the options in section 26. Given an image and a natural-language
expression naming one object in it, predict the mask of that object alone.

| | Modality | Representation |
| --- | --- | --- |
| A | RGB image | 3 channels, resized to a fixed square |
| B | Referring expression | token sequence, mean 3.50 tokens |

Output is a binary mask over the referred object. The prediction unit is one (image, expression)
pair, so a single image yields several independent predictions with different targets.

**Pairing linkage.** The expression and the mask describe the same annotated COCO instance, joined
by `ann_id`; the image is the frame that instance was annotated in. This is entity-level linkage
within one source dataset, not two corpora paired after the fact. Every `ann_id` referenced by the
50,000 refs resolves in `instances.json`; none is missing.

## Paired samples and annotation types

| | Count |
| --- | ---: |
| Images | 19,994 |
| Referred objects, each a COCO polygon instance | 50,000 |
| Referring expressions, each paired with its object's mask | 142,210 |
| Expressions per object | 2.84 |

Annotations are COCO-style polygons, rasterised to binary masks with `pycocotools`. All 50,000
referred objects carry a polygon; none is stored as RLE and none is absent.

The widely quoted figure for this dataset is 142,209 expressions. The file holds 142,210.

## Preliminary pair-quality and distribution analysis

**Missing modalities: none.** Across all 50,000 refs, 0 lack an expression, 0 reference an absent
image, and 0 reference an absent annotation. Section 29.2's missing-modality handling is therefore
not applicable, and the report will say so rather than invent a case.

**The text is necessary, by construction.** No referred object is the only instance of its
category in its image: the annotation game only used images holding two or more instances of the
same category. A referred object competes with 4.90 same-category distractors on average, median
3, maximum 22.

| Same-category instances in the image | Objects | Share |
| ---: | ---: | ---: |
| 2 | 16,918 | 33.8% |
| 3 | 8,708 | 17.4% |
| 4 | 5,863 | 11.7% |
| 5 | 3,703 | 7.4% |
| 6 or more | 14,808 | 29.6% |

This is the property the whole assignment rests on. An image-only baseline cannot identify the
target better than chance among its distractors, so any gap it leaves is attributable to the
language rather than to a weak vision model.

**The text is short.** Mean 3.50 tokens, median 3, 90th percentile 6, maximum 39, over a
vocabulary of 10,341 types in 497,738 tokens. 36.8% of expressions are one or two tokens. The ten
most frequent tokens are *left, right, on, in, the, guy, man, shirt, front, of*: spatial language
dominates, which is expected since the collection game forbade pointing and rewarded fast
disambiguation.

**The categories are concentrated.** 78 of COCO's 80 categories appear, but person alone accounts
for 49.5% of referred objects; no other category reaches 2.2%.

| Category | Objects | Share |
| --- | ---: | ---: |
| person | 24,767 | 49.5% |
| elephant | 1,070 | 2.1% |
| zebra | 1,053 | 2.1% |
| chair | 1,024 | 2.0% |
| giraffe | 1,010 | 2.0% |
| 73 others | 21,076 | 42.2% |

**The targets are large.** The median referred object covers 8.3% of its frame, quartiles 5.4%
and 14.1%, and not one of the 50,000 falls below 1%. Thin-structure error, the usual difficulty in
segmentation, is therefore not the binding constraint here: the difficulty is choosing which of
several same-category instances the expression names.

**Annotation noise.** An independent re-annotation effort, Ref-L4, reviewed 21,586 RefCOCO
annotations and judged 3,054 of them erroneous, a 14% rate. Those errors are the natural source of
the conflicting-evidence cases section 30 requires, and the error analysis will sample from them
rather than assume the labels are clean.

**Images are not uniform.** 1,052 distinct resolutions occur among the referred images, the most
common being 640x480 in 12,793 cases. Every batch therefore needs resizing to a common size, and
masks must be resized by nearest neighbour, since interpolating a binary mask produces values
belonging to neither class.

## Data-splitting plan and leakage-prevention unit

**The leakage-prevention unit is the image.** All expressions and all referred objects belonging
to one image stay in one partition, which satisfies the rule that an entity's records are never
divided. The UNC partition already enforces this and is adopted unchanged, materialised as
[`splits/refcoco-unc.json`](splits/refcoco-unc.json) so every run reads the same partition and a
reported number traces to it without re-downloading the archive.

| Partition | Objects | Images | Expressions |
| --- | ---: | ---: | ---: |
| train | 42,404 | 16,994 | 120,624 |
| val | 3,811 | 1,500 | 10,834 |
| testA | 1,975 | 750 | 5,657 |
| testB | 1,810 | 750 | 5,095 |

Measured, not assumed: the image sets of all six partition pairs are disjoint, with zero overlap
in every case. testA holds only people and testB only non-people, so the pair measures whether a
model has learned the dominant person category at the expense of the tail. Model selection uses
val; testA and testB are read once, after selection concludes, so neither influences a decision.

## Evaluation metrics

mIoU is primary, with Dice secondary, matching section 29.3's "mIoU as appropriate" for
segmentation. Both are computed per (image, expression) pair over the foreground and background
classes, then averaged over the split.

Precision at an IoU threshold, conventional in this literature, is recorded alongside but decides
nothing, because a single threshold hides where a model sits in the distribution.

## Baselines and planned models

Section 29.1 requires four, and the distractor analysis above makes all four meaningful.

| | Model | What it tests |
| --- | --- | --- |
| 1 | Image only | Segments a salient instance with no access to the expression. Bounded by the distractor count: it cannot choose among same-category instances |
| 2 | Text only | Predicts a mask from the expression alone, as a spatial and size prior. Weak by construction; it measures how much of the task language alone determines |
| 3 | Simple fusion | Frozen text encoder, pooled sentence embedding broadcast and concatenated onto the visual feature map, late fusion |
| 4 | Improved fusion | Cross-attention between token embeddings and visual positions, so individual words can address individual regions |

The vision branch is DeepLabV3 with a MobileNetV3-Large backbone, re-headed for one foreground
class, starting from the published COCO-pretrained weights, which were fitted on the same image
distribution these expressions describe.

The text encoder is a pretrained sentence encoder, frozen in model 3 and fine-tuned in model 4;
expressions average 3.50 tokens, so its cost is small beside the vision branch.

The mandatory ablation removes modality B from model 4 and, separately, replaces cross-attention
with the concatenation of model 3, which isolates the modality from the fusion design.

| Component | Origin |
| --- | --- |
| Fusion modules, text branch, mask rasterisation, dataset | Written from scratch |
| DeepLabV3 vision branch | `torchvision.models.segmentation` |
| Text encoder | pretrained checkpoint, Hugging Face |
| Polygon to mask conversion | `pycocotools` |
| Metrics, trainer, figures, run tracking | `shared/`, the package every assignment installs |

## Compute estimate

Measured by [`notebooks/compute.ipynb`](notebooks/compute.ipynb) on an Apple M4 Pro, 14 cores,
48 GB, at a 320 square and batch size 8, and written to
[`results/compute.json`](results/compute.json). A training step is forward, loss and backward;
inference is forward only, without autograd.

| Model | Trainable | s/step | ms/pair | Peak | Hours per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image only | 11,020,594 | 0.108 | 4.1 | 2.23 GB | 3.17 |
| Text only | 67,937,792 | 0.041 | 0.6 | 2.23 GB | 1.20 |
| Simple fusion | 4,005,170 | 0.101 | 3.4 | 2.24 GB | 2.96 |
| Improved fusion | 73,468,466 | 0.146 | 3.8 | 3.27 GB | 4.30 |

One run is 20 epochs over 42,404 pairs, 5,300 steps per epoch under the subset rule below.

Two measurements are worth stating because they are not what the parameter counts suggest.
Simple fusion carries the fewest trainable parameters of the four, 4.0M, because its text
encoder is frozen, yet it costs almost as much per step as the image-only baseline: a frozen
encoder still runs its forward pass. And the improved model's 73.5M trainable parameters cost
only 36% more per step than the 11.0M of the image-only baseline, because expressions average
3.50 tokens and attention over so short a sequence is cheap beside the vision branch.

| Runs | Seeds | Hours |
| --- | ---: | ---: |
| Image only | 1 | 3.17 |
| Text only | 1 | 1.20 |
| Simple fusion | 3 | 8.88 |
| Improved fusion | 3 | 12.89 |
| **Total** | | **26.14** |

Three seeds for the two models the decision rule must separate, one for each baseline, which the
rule already treats as indicative. The total fits the one-week draft and final windows.

The Apple GPU backend has no scaled-dot-product kernel that supports dropout, so attention runs
on the eager path here while CUDA would use the fused one. Timings therefore compare with each
other but do not transfer between backends.

Storage is 184 MB of annotations unpacked, plus 3.3 GB for the 19,994 referenced images against
13.5 GB for the whole COCO 2014 training archive.

## Subset rules

Two, both declared in advance.

**One expression per object during training**, sampled uniformly per epoch from that object's
2.84 expressions. This cuts the epoch to a third without discarding any object, and every
expression is still seen across epochs. Evaluation uses all expressions, so no reported number
depends on the rule.

**No subsetting of images or objects.** The full UNC partition is used, so the split and its
disjointness guarantee carry over unchanged.
