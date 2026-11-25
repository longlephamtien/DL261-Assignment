# Assignment 3 experiment plan

This plan tests whether the referring expression contributes information beyond the
RGB image, and whether token-level cross-attention uses that information better than
a pooled late-fusion representation.

## Common protocol

All experiments use the RefCOCO UNC split and the committed image-level split in
[`splits/refcoco-unc.json`](splits/refcoco-unc.json). All expressions and target
objects from one image remain in the same partition. Training uses one expression
sampled per object per epoch; validation and test use every expression. No test result
is used for model selection.

Unless an experiment explicitly changes a factor below, it fixes:

- input: 320 x 320 RGB image, tokenized with `distilbert-base-uncased`;
- loader: batch size 8, four workers, the declared horizontal-flip augmentation with
  left/right expression rewriting;
- optimization: AdamW, learning rate `3e-4`, weight decay `0.05`, cosine schedule,
  combined cross-entropy plus Dice loss, 20 epochs, and early stopping patience 5;
- split and evaluation: train for fitting, val for selection, and one final evaluation
  on testA (people-heavy) and testB (non-person); and
- reporting: mean per-pair IoU (mIoU, primary), mean per-pair Dice (secondary), and
  precision at IoU >= 0.5. Report mean and standard deviation across seeds for the
  fusion models; report the single measured value for the one-seed baselines.

The primary comparison is mIoU. Dice and precision@0.5 are supporting metrics, and
pooled pixel IoU is retained only as a diagnostic because it is dominated by large
objects. Results are also stratified by testA/testB and by same-category distractor
count when predictions are available.

## Planned experiments

| ID | Hypothesis | Changed factor | Fixed factors | Metrics | Conclusion criterion |
| --- | --- | --- | --- | --- | --- |
| E0 | The data and evaluation pipeline are deterministic and preserve the pairing contract. | None in the model; run split, loader, shape, and metric checks. | All dataset and protocol settings. | Split overlap, paired counts, target-mask validity, mIoU/Dice unit-test outputs. | Pass only if image sets are disjoint, every selected pair has image/text/mask, and metric checks match their hand-computed values. |
| E1-A | The image contains useful object and boundary information, but cannot reliably identify the referred instance among same-category distractors. | Remove modality B: image-only DeepLabV3-MobileNetV3. | Vision preprocessing, loss, optimizer, epochs, split, and evaluation. | Val/testA/testB mIoU, Dice, precision@0.5; distractor-count slices. | The baseline is valid if it trains and produces non-trivial masks; its gap to fusion estimates the value of language. |
| E1-B | A short expression supplies spatial and size priors but cannot reconstruct the exact mask without pixels. | Remove modality A: text-only DistilBERT mask predictor. | Text tokenizer, loss, optimizer, epochs, split, and evaluation. | Same metrics and slices as E1-A. | The baseline is valid if it trains; substantially lower mIoU than image-only or fusion supports the claim that text alone is insufficient. |
| E2 | Combining both modalities should improve referred-instance selection over either unimodal baseline. | Add both modalities with pooled sentence embedding broadcast and concatenated at the visual feature map (simple late fusion). | Same split, input size, loss, optimizer, epochs, and evaluation; text encoder frozen. | Same metrics, with mean +/- std over seeds 0, 1, and 2. | Fusion is useful if val mIoU exceeds both unimodal baselines by at least 1 percentage point and the direction is reproduced in at least 2/3 seeds. |
| E3 | Letting each visual position attend to the expression tokens should resolve spatial words and distractors better than pooled fusion. | Replace late concatenation with visual-to-text cross-attention; fine-tune the text encoder. | Same vision backbone, input pipeline, split, loss, optimizer, epochs, and evaluation; 8 attention heads. | Same metrics and seeds as E2; also compare testA versus testB. | Improved fusion is preferred if mean val mIoU is at least 2 percentage points above E2, with no more than a 1-point drop on either test split. |
| E4 (ablation) | The language modality is necessary for selecting the referred instance, not merely for regularization. | Remove text from the improved-fusion design and use the image-only model. | Keep the vision encoder, decoder/head, input pipeline, loss, optimizer, epochs, split, and evaluation fixed. | Delta mIoU, Dice, and precision@0.5: improved fusion minus image-only, overall and by distractor count. | The modality ablation supports the hypothesis if removing text lowers mIoU on val and on both testA and testB; a larger drop for more distractors is additional evidence. |
| E5 (ablation) | Token-level interaction, rather than simply adding more parameters, is responsible for the improved result. | Replace cross-attention with the simple late-concat fusion while retaining the same two modalities. | Same image/text encoders, optimizer, loss, split, evaluation, and seed set; only fusion operation is changed. | Delta mIoU, Dice, and precision@0.5 between E3 and E2, including seed standard deviation. | The fusion ablation supports the hypothesis if E3 beats E2 on val mIoU and the improvement is reproduced on at least one held-out test split without a >1-point regression on the other. |
| E6 | The selected fusion does not only exploit the dominant person category. | No training change; evaluate the selected checkpoint separately on testA and testB. | Selected model, checkpoint, split, and all preprocessing. | mIoU, Dice, precision@0.5 on testA and testB; category/distractor slices. | Generalization is credible if the selected fusion beats the relevant unimodal baseline on both testA and testB; report any category-specific failure rather than pooling it away. |

E4 and E5 are the required ablations: E4 removes a modality and E5 changes the
fusion component. They are not extra model families: E4 is the planned image-only
baseline and E5 is the planned simple-fusion baseline, each compared directly with
the improved-fusion model under the same protocol.

## Decision and reporting rules

The improved model is selected using validation mIoU only. If two configurations are
within 1 percentage point, prefer the lower-variance configuration; if still tied,
prefer the cheaper configuration. After selection, run the selected checkpoint once
on testA and testB and report both separately. Conclusions must state whether each
hypothesis was supported, mixed, or not supported; a non-significant or inconsistent
change is not described as an improvement.

The plan does not tune on testA/testB, change the split after seeing results, or claim
statistical significance from three seeds alone. Seed-level values, checkpoint epoch,
wall time, parameter count, and failed/early-stopped runs are retained with each
result.

## Compute and schedule

The measured estimates in [`results/compute.json`](results/compute.json) are for 20
epochs, 42,404 training pairs per epoch, batch size 8, and the common 320-square
input:

| Run group | Seeds | Estimated hours |
| --- | ---: | ---: |
| E1-A image-only | 1 | 3.17 |
| E1-B text-only | 1 | 1.20 |
| E2 simple fusion | 3 | 8.88 |
| E3 improved fusion | 3 | 12.89 |
| **Total planned training** |  | **26.14** |

E4 and E5 are paired comparisons using the E1-A/E2/E3 runs, so they add no training
hours. E0, E6, validation, final inference, and figure generation are lightweight
relative to training and remain inside the one-week M2/M3 windows. Main experiments
start only after the proposal approval gate is recorded.
