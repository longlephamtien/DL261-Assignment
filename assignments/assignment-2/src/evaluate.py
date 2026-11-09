"""Metrics, latency, and prediction overlays for one finished run.

    python -m src.evaluate <run_id> [--split val]

Validation is the default because Cityscapes withholds the test labels. Output goes to
`results/evaluation/<run_id>/<split>/` and is published with the run. Implemented by #63.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from shared import io as shared_io
from shared.device import count_parameters, peak_memory, reset_peak_memory, select_device, synchronize
from shared.figures import save

from .data import IMAGENET_NORMALIZATION, build_dataloaders
from .interfaces import (
    CLASS_NAMES, EVALUATION_SPLITS, IGNORE_INDEX, NUM_CLASSES, EvaluationReport, RunSummary, Split,
)
from .metrics import ConfusionMatrix
from .models import build_model
from .utils import evaluation_dir, report_path, run_dir as resolve_run


def load_checkpoint(directory: Path) -> tuple[dict, RunSummary, nn.Module]:
    """Rebuild the model of a run and restore its selected weights."""
    config = shared_io.read_yaml(directory / "config.yaml")
    summary: RunSummary = shared_io.read_json(directory / "summary.json")
    model = build_model(config["model"]["name"], **config["model"].get("args", {}))
    model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location="cpu"))
    return config, summary, model


def image_miou(target: Tensor, prediction: Tensor) -> float:
    """Mean IoU of one image, over the classes that image actually contains."""
    matrix = ConfusionMatrix()
    matrix.update(target.unsqueeze(0), prediction.unsqueeze(0))
    return float(matrix.mean_iou())


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[ConfusionMatrix, list[float]]:
    """The confusion matrix over a split, and the mIoU of each image in loader order.

    Masks are never concatenated across the split: at 512 by 512 a Cityscapes validation pass
    would hold hundreds of millions of pixels, so each batch leaves behind only the running
    matrix and one float per image.
    """
    model.eval().to(device)
    matrix = ConfusionMatrix()
    scores: list[float] = []

    for images, targets in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        matrix.update(targets, predictions)
        scores.extend(image_miou(targets[index], predictions[index]) for index in range(targets.shape[0]))

    return matrix, scores


def select_examples(scores: list[float], count: int) -> list[tuple[int, str]]:
    """Image indices spanning the mIoU range, each named by the case it illustrates.

    #74 asks for correct, difficult-but-correct and failure cases, so the split is ranked by
    per-image mIoU and sampled at even ranks across the whole of it: the best image is the
    correct case, the worst the failure, and the ranks between them the difficult ones.
    """
    if not scores or count < 1:
        return []

    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    count = min(count, len(order))
    chosen = [order[round(rank * (len(order) - 1) / max(count - 1, 1))] for rank in range(count)]
    tiers = ["correct"] if count == 1 else ["correct"] + ["difficult"] * (count - 2) + ["failure"]
    return list(zip(chosen, tiers[:count]))


def export_overlays(images: Tensor, targets: Tensor, predictions: Tensor, tiers: list[str], directory: Path,
                    normalization: tuple[list[float], list[float]] = IMAGENET_NORMALIZATION) -> None:
    """One figure per example: image, ground truth, prediction, and the pixels that differ."""
    records = []
    for position, tier in enumerate(tiers):
        score = image_miou(targets[position], predictions[position])
        figure = _overlay_figure(
            images[position], targets[position], predictions[position], score, tier, normalization
        )
        save(figure, directory / f"example_{position + 1}_{tier}")
        records.append({"rank": position + 1, "tier": tier, "miou": score})

    shared_io.write_json(directory / "examples.json", records)


def _overlay_figure(image: Tensor, target: Tensor, prediction: Tensor, score: float, tier: str,
                    normalization: tuple[list[float], list[float]]) -> plt.Figure:
    """Image, ground truth and prediction side by side, with the pixels that differ marked."""
    picture = _denormalize(image, normalization)
    truth, guess = _colourize(target), _colourize(prediction)
    wrong = ((target != prediction) & (target != IGNORE_INDEX)).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 16 / 4 * (image.shape[-2] / image.shape[-1]) + 0.5))
    for ax, panel, title in (
        (axes[0], picture, "image"),
        (axes[1], truth, "ground truth"),
        (axes[2], guess, "prediction"),
    ):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9)

    axes[3].imshow(picture)
    axes[3].imshow(np.ma.masked_where(~wrong, wrong), cmap="autumn", alpha=0.65, interpolation="nearest")
    axes[3].set_title(f"errors, {wrong.mean() * 100:.1f}% of labelled pixels", fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{tier} case, image mIoU {score:.3f}")
    return fig


def _denormalize(image: Tensor, normalization: tuple[list[float], list[float]]) -> np.ndarray:
    """Invert the loader's normalization, so the panel shows the photograph and not the tensor."""
    mean, std = (torch.tensor(stat).view(3, 1, 1) for stat in normalization)
    return (image * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()


def _colourize(mask: Tensor) -> np.ndarray:
    """A trainId mask as an RGB image, with unlabelled pixels black."""
    palette = (plt.get_cmap("tab20")(np.arange(NUM_CLASSES) % 20)[:, :3] * 255).astype(np.uint8)
    array = mask.numpy()
    output = np.zeros((*array.shape, 3), dtype=np.uint8)
    labelled = array != IGNORE_INDEX
    output[labelled] = palette[array[labelled]]
    return output


@torch.no_grad()
def measure_latency(model: nn.Module, device: torch.device, crop: list[int], inference: dict) -> dict[str, float]:
    """Milliseconds per image at each configured batch size, after warm-up, on one device.

    Measured without autograd: a graph kept across the timed iterations is not part of
    inference, and on CUDA its activations would dominate the peak memory reported beside it.
    """
    latency = {}
    for batch_size in inference["batch_sizes"]:
        batch = torch.randn(batch_size, 3, *crop, device=device)
        for _ in range(inference["warmup_batches"]):
            model(batch)
        synchronize(device)

        start = time.perf_counter()
        for _ in range(inference["timed_batches"]):
            model(batch)
        synchronize(device)

        elapsed = time.perf_counter() - start
        latency[str(batch_size)] = elapsed * 1000 / (inference["timed_batches"] * batch_size)
    return latency


def evaluate(run: str | Path, split: Split = "val") -> EvaluationReport:
    """Score one run on a split and write its report and overlays."""
    directory = resolve_run(run)
    config, summary, model = load_checkpoint(directory)
    device = select_device()
    settings = config["evaluation"]

    reset_peak_memory(device)
    loaders = build_dataloaders(config)
    matrix, scores = predict(model, loaders[split], device)
    iou, dice = matrix.iou(), matrix.dice()

    report: EvaluationReport = {
        "run_id": summary["run_id"],
        "model": summary["model"],
        "backbone": summary["backbone"],
        "split": split,
        "miou": matrix.mean_iou(),
        "dice": matrix.mean_dice(),
        "pixel_accuracy": matrix.pixel_accuracy(),
        "per_class_iou": dict(zip(CLASS_NAMES, iou.tolist())),
        "per_class_dice": dict(zip(CLASS_NAMES, dice.tolist())),
        "confusion_matrix": matrix.matrix.tolist(),
        "parameters": count_parameters(model),
        "training_seconds": summary["training_seconds"],
        "inference_ms_per_image": measure_latency(model, device, config["input"]["crop"], settings["inference"]),
        "peak_memory_bytes": peak_memory(device),
    }

    shared_io.write_json(report_path(config, summary["run_id"], split), report)
    chosen = select_examples(scores, settings["example_predictions"])
    if chosen:
        images, targets, predictions = _gather(model, loaders[split].dataset, [index for index, _ in chosen], device)
        export_overlays(
            images, targets, predictions, [tier for _, tier in chosen],
            evaluation_dir(config, summary["run_id"], split) / "examples",
            tuple(config["dataset"].get("normalization", IMAGENET_NORMALIZATION)),
        )
    return report


@torch.no_grad()
def _gather(model: nn.Module, dataset, indices: list[int], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Re-read only the chosen images and predict them again, so the split is never held."""
    images = torch.stack([dataset[index][0] for index in indices])
    targets = torch.stack([dataset[index][1] for index in indices])
    return images, targets, model(images.to(device)).argmax(dim=1).cpu()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score one run and write its evaluation report.")
    parser.add_argument("run", type=resolve_run, help="run id, or a path to the run directory")
    parser.add_argument("--split", choices=EVALUATION_SPLITS, default="val", help="test only after model selection")
    args = parser.parse_args(argv)

    report = evaluate(args.run, args.split)
    print(f"{report['model']:<12} {args.split:<5} mIoU {report['miou']:.4f}  Dice {report['dice']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
