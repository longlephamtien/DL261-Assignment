"""Model contract: shapes, registration, and what the pretrained wrapper must record."""

from __future__ import annotations

import pytest
import torch

from src.interfaces import NUM_CLASSES
from src.models import build_model, list_models
from src.models.deeplabv3 import BACKBONES, DeepLabV3Segmenter, normalization_for


def untrained(**kwargs) -> DeepLabV3Segmenter:
    """Built without downloading weights; the wiring under test does not depend on them."""
    return build_model("deeplabv3", pretrained=False, **kwargs)


def test_deeplabv3_is_registered() -> None:
    assert "deeplabv3" in list_models()


@pytest.mark.parametrize("size", [64, 128])
def test_forward_returns_per_pixel_logits_at_the_input_resolution(size: int) -> None:
    """The evaluation interface takes argmax over dim 1, so the head must keep the input size."""
    model = untrained().eval()
    output = model(torch.randn(2, 3, size, size))

    assert output.shape == (2, NUM_CLASSES, size, size)
    assert output.dtype == torch.float32


def test_forward_returns_a_tensor_not_the_torchvision_dictionary() -> None:
    """torchvision returns an ordered dict; the registry contract is a plain tensor."""
    assert isinstance(untrained().eval()(torch.randn(1, 3, 64, 64)), torch.Tensor)


def test_the_auxiliary_head_is_dropped_rather_than_computed() -> None:
    assert untrained().network.aux_classifier is None


def test_the_head_is_sized_to_the_cityscapes_label_set() -> None:
    """The published weights predict 21 classes; keeping that head would score the wrong labels."""
    assert untrained().network.classifier[-1].out_channels == NUM_CLASSES


def test_an_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported backbone"):
        build_model("deeplabv3", backbone="resnet18", pretrained=False)


def test_every_declared_backbone_resolves() -> None:
    """A typo in the table would only surface once a run started.

    The builders are checked without instantiating them: allocating ResNet-101 costs more
    than the rest of this suite put together, and a broken entry shows up here anyway.
    """
    for backbone, (builder, weights_enum) in BACKBONES.items():
        assert callable(builder), backbone
        assert hasattr(weights_enum, "DEFAULT"), backbone


def test_freezing_leaves_only_the_head_trainable() -> None:
    model = untrained(freeze_backbone=True)

    assert not any(p.requires_grad for p in model.network.backbone.parameters())
    assert all(p.requires_grad for p in model.network.classifier.parameters())


def test_an_unfrozen_model_trains_end_to_end() -> None:
    model = untrained()
    assert all(p.requires_grad for p in model.network.backbone.parameters())


def test_the_weight_source_is_recorded() -> None:
    """Section 4.2 requires the provenance of pretrained weights to be traceable."""
    model = untrained()
    assert model.weights_source == "random initialisation"
    assert model.pretrained is False
    assert model.normalization is None, "no pretrained weights means no inherited preprocessing"


def test_the_pretrained_preprocessing_is_read_from_the_weights() -> None:
    """Normalising differently than the published weights were trained with degrades them."""
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as Weights

    declared = normalization_for(Weights.DEFAULT)
    expected = Weights.DEFAULT.transforms()
    assert declared["mean"] == list(expected.mean)
    assert declared["std"] == list(expected.std)
    assert declared["source"] == Weights.DEFAULT.name


def test_the_loader_default_matches_what_the_pretrained_weights_expect() -> None:
    """data.py keeps its own copy; a divergence would degrade the pretrained backbone silently."""
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as Weights

    from src.data import IMAGENET_NORMALIZATION

    declared = normalization_for(Weights.DEFAULT)
    assert list(IMAGENET_NORMALIZATION[0]) == declared["mean"]
    assert list(IMAGENET_NORMALIZATION[1]) == declared["std"]


def test_gradients_reach_the_head() -> None:
    """A forward pass that produces no gradient would train nothing."""
    model = untrained()
    model(torch.randn(2, 3, 64, 64)).sum().backward()

    head_weight = model.network.classifier[-1].weight
    assert head_weight.grad is not None and head_weight.grad.abs().sum() > 0
