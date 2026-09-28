# Assignment 1 experiment plan

## Protocol

| Element | Value |
| --- | --- |
| Split | 54,000 train, 6,000 validation, 10,000 test; stratified, seed 0, committed |
| Normalization | mean 0.2858, standard deviation 0.3529, training split only |
| Augmentation | random crop with edge padding 2, then horizontal flip, training loader only |
| Optimization | AdamW, learning rate 3e-4, weight decay 0.05, batch size 128 |
| Schedule | cosine annealing over 30 epochs, early stopping after 5 |
| Checkpoint | highest validation macro-F1 |
| Metrics | macro-F1 primary; accuracy, parameters, seconds per epoch, inference latency secondary |
| Environment | one machine per comparison; hardware, library versions and commit recorded per run |

A run that deviates from the protocol is repeated rather than reported. The test split is evaluated once, after model selection concludes.

## Seeds and budget

Every reported configuration runs at seeds 0, 1 and 2, and is reported as a mean plus or minus one standard deviation. Tuning is capped at six configurations per model, judged on validation macro-F1 at seed 0 and varying only `model.args`.

## Decision rule

Model A is reported as superior to model B only when its mean validation macro-F1 is higher and the two intervals of mean plus or minus one standard deviation do not overlap. Otherwise the two are reported as indistinguishable under this protocol, which is itself a result. Accuracy alone never decides a comparison.

## Experiments

**E1, architecture families.** *Hypothesis:* architectures encoding a spatial prior outperform those treating the image as an unordered vector at comparable capacity. *Varies:* model family across linear, MLP, convolutional, recurrent and Transformer. *Fixed:* the protocol and the tuning budget. *Metrics:* validation macro-F1, accuracy, parameter count, seconds per epoch, inference latency at batch sizes 1 and 128.

**E2, sequence representation.** *Hypothesis:* patch tokens outperform row and column tokens for the recurrent and Transformer models, because a patch preserves two-dimensional locality. *Varies:* `input.representation` across rows, columns and patches, at patch size 4. *Fixed:* architecture, its arguments, and the protocol. *Metrics:* validation macro-F1, related to sequence length and features per token.

**E3, recurrent cell.** *Hypothesis:* GRU matches LSTM at lower capacity, since sequences of 28 or 49 steps are short enough that the additional gate contributes little. *Varies:* cell type. *Fixed:* representation, hidden size, depth, dropout, and the protocol. *Metrics:* validation macro-F1, parameter count, seconds per epoch. Where the two are indistinguishable, the smaller model is preferred and the margin stated.

**E4, matched parameter budget.** *Hypothesis:* the E1 ranking reflects architecture rather than capacity and therefore survives equalised parameter counts. *Varies:* model family, with width tuned to within 10 percent of a shared budget. *Fixed:* the budget and the protocol. *Metrics:* validation macro-F1, with realised parameter counts reported so the tolerance is auditable. An extension; it cannot substitute for E1.