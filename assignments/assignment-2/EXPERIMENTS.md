# Assignment 2 experiment plan

## Protocol

| Element | Value |
| --- | --- |
| Split | 2,406 train, 569 validation, 500 test; city-level, seed 0, committed |
| Input | resized to 512×1024; training loader takes a random 512×512 crop, evaluation the full resized image |
| Normalization | ImageNet statistics, identical for every model |
| Augmentation | random crop, then horizontal flip with probability 0.5, training loader only |
| Optimization | AdamW, learning rate 3e-4, weight decay 0.05, batch size 8 |
| Schedule | cosine annealing over 40 epochs, early stopping after 8 |
| Loss | cross-entropy with `ignore_index=255`, unless the experiment varies it |
| Checkpoint | highest validation mIoU |
| Metrics | mIoU primary; Dice, per-class IoU, trainable parameters, seconds per epoch, inference latency secondary |
| Environment | one machine per comparison; hardware, library versions and commit recorded per run |

A run that deviates from the protocol is repeated rather than reported. The test split, the official validation partition, is evaluated once per configuration, after model selection concludes.

## Seeds and budget

Every configuration entering a comparison runs at seeds 0, 1 and 2, and is reported as a mean plus or minus one standard deviation. The seed changes initialisation, crop positions and batch order; the split stays the seed 0 file for every run.

No hyperparameter search is run. Every configuration uses the shared protocol and the committed `configs/<model>.yaml`, and an experiment changes exactly one key through `--set`, so the search effort is zero and identical across arms.

Planned cost at the 512 crop, from the timings in [PROPOSAL.md](PROPOSAL.md), as upper bounds before early stopping:

| Runs | Seeds | Hours |
| --- | ---: | ---: |
| U-Net, base width 32, depth 4 | 3 | 7.3 |
| DeepLabV3-MobileNetV3, full fine-tune, cross-entropy | 3 | 2.5 |
| DeepLabV3-MobileNetV3, frozen backbone | 3 | 2.5 |
| DeepLabV3-MobileNetV3, Dice loss | 3 | 2.5 |
| DeepLabV3-ResNet50, full fine-tune | 1 | 5.8 |
| **Total** | | **20.6** |

The frozen arm is bounded by the full-fine-tune timing, since it computes the same forward pass and fewer gradients. The total exceeds the proposal's 16.4 hours because the frozen arm and the loss ablation now run at three seeds, which the decision rule requires.

## Decision rule

Configuration A is reported as superior to configuration B only when its mean validation mIoU is higher and the two intervals of mean plus or minus one standard deviation do not overlap. Otherwise the two are reported as indistinguishable under this protocol, which is itself a result. Pixel accuracy never decides a comparison, since predicting road, building and vegetation alone already reaches about 0.76.

A claim about a single class applies the same rule to that class's validation IoU. A configuration run at one seed supports no claim of superiority and is reported as indicative.

## Experiments

**E1, baseline against pretrained model.** *Hypothesis:* the pretrained DeepLabV3-MobileNetV3 outperforms the U-Net trained from scratch at comparable capacity, 11.0M against 7.8M trainable parameters, because 2,406 training images are too few to learn general features from scratch. *Varies:* model, between `configs/unet.yaml` and `configs/deeplabv3.yaml`. *Fixed:* the protocol, including loss and normalization. *Metrics:* validation mIoU, Dice and per-class IoU, trainable parameters, seconds per epoch, inference latency at batch sizes 1 and 8.

**E2, frozen against fully fine-tuned backbone, the controlled experiment.** *Hypothesis:* fine-tuning every parameter outperforms training the head on a frozen backbone, because ImageNet features are not adapted to street scenes; the gap is widest on thin classes such as pole and traffic sign, whose boundaries depend on low-level features the frozen arm cannot change. *Varies:* `model.args.freeze_backbone`, false against true. *Fixed:* backbone, pretrained weights, head, learning rate and the rest of the protocol. *Metrics:* validation mIoU and per-class IoU, trainable parameters, seconds per epoch. Where the two are indistinguishable, the frozen arm is preferred for its lower training cost and the margin stated. Freezing sets `requires_grad` to false but leaves the backbone in training mode, so its batch-normalization statistics still adapt to Cityscapes; the frozen arm therefore fixes every learned weight but not the running statistics, and the report states this.

**E3, cross-entropy against Dice loss, the ablation.** *Hypothesis:* Dice loss raises IoU on classes that are rare but large when present, namely train, bus and truck, because it weights every class equally regardless of pixel count; it does not help thin classes, whose errors lie at boundaries rather than in how often the class is seen. *Varies:* `train.loss`, `cross_entropy` against `dice`. *Fixed:* DeepLabV3-MobileNetV3, fully fine-tuned, and the protocol. *Metrics:* validation mIoU and Dice, and per-class IoU for train, bus, truck, pole and traffic sign. The hypothesis holds only if the per-class rule separates the two arms on at least two of train, bus and truck.

**E4, backbone.** *Hypothesis:* ResNet50 outperforms MobileNetV3-Large at 3.6 times the parameters and 7 times the cost per step. *Varies:* `model.args.backbone`. *Fixed:* full fine-tuning, cross-entropy, and the protocol. *Metrics:* validation mIoU, trainable parameters, seconds per epoch, inference latency. An extension run at seed 0 only, so it is reported as indicative and cannot substitute for E2 or E3.

E1 maps to issues #65 and #71, E2 to E4 to #72, and the cost columns to #73.
