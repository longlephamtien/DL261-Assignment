"""Metrics, latency, and prediction examples for one finished run.

    python -m src.evaluate <run_id> [--split val]

Validation is the default; the test split is evaluated once, after model selection. Output
goes to `results/evaluation/<run_id>/<split>/` and is published with the run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import Tensor, nn
from torch.utils.data import DataLoader

from shared import io as shared_io
from shared.device import synchronize

from .data import build_dataloaders
from .interfaces import (
    CLASS_NAMES,
    EVALUATION_SPLITS,
    IMAGE_SHAPE,
    EvaluationReport,
    RunSummary,
    Split,
    count_parameters,
    select_device,
)
from .models import build_model
from .utils import evaluation_dir, report_path, run_dir as resolve_run


def load_checkpoint(run_dir: Path) -> tuple[dict, RunSummary, nn.Module]:
    config = shared_io.read_yaml(run_dir / "config.yaml")
    summary: RunSummary = shared_io.read_json(run_dir / "summary.json")
    model = build_model(config["model"]["name"], **config["model"].get("args", {}))
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=select_device()))
    return config, summary, model.to(select_device()).eval()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Targets, predictions, confidences, and images, in loader order."""
    targets, predictions, confidences, images = [], [], [], []
    for batch_images, batch_targets in loader:
        probabilities = model(batch_images.to(device)).softmax(dim=1).cpu()
        confidence, prediction = probabilities.max(dim=1)
        targets.append(batch_targets)
        predictions.append(prediction)
        confidences.append(confidence)
        images.append(batch_images)
    return torch.cat(targets), torch.cat(predictions), torch.cat(confidences), torch.cat(images)


@torch.no_grad()
def measure_latency(model: nn.Module, device: torch.device, inference: dict) -> dict[str, float]:
    """Milliseconds per image at each configured batch size, after warm-up, on one device."""
    latency = {}
    for batch_size in inference["batch_sizes"]:
        batch = torch.randn(batch_size, *IMAGE_SHAPE, device=device)
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


def export_examples(
    images: Tensor,
    targets: Tensor,
    predictions: Tensor,
    confidences: Tensor,
    normalization: dict,
    count: int,
    directory: Path,
) -> None:
    """One grid of correct predictions and one of the most confident errors."""
    correct = predictions == targets
    chosen = {
        "correct": correct.nonzero().flatten()[:count],
        "incorrect": (~correct).nonzero().flatten()[confidences[~correct].argsort(descending=True)][:count],
    }

    directory.mkdir(parents=True, exist_ok=True)
    for name, indices in chosen.items():
        if not len(indices):
            continue
        figure, axes = plt.subplots(1, len(indices), figsize=(1.5 * len(indices), 2.1))
        for axis, index in zip(axes.flat if len(indices) > 1 else [axes], indices):
            pixels = images[index, 0] * normalization["std"] + normalization["mean"]
            axis.imshow(pixels.clamp(0, 1).numpy(), cmap="gray", interpolation="nearest")
            axis.set_title(
                f"{CLASS_NAMES[targets[index]]}\n{CLASS_NAMES[predictions[index]]} {confidences[index]:.2f}",
                fontsize=7,
            )
            axis.axis("off")
        figure.suptitle(f"{name}, true label above predicted label and confidence", fontsize=8)
        figure.tight_layout()
        figure.savefig(directory / f"{name}.png", dpi=200)
        plt.close(figure)


def evaluate(run: str | Path, split: Split = "val") -> EvaluationReport:
    """Score one run on a split and write its report and example grids."""
    run_dir = resolve_run(run)
    config, summary, model = load_checkpoint(run_dir)
    device = select_device()
    settings = config["evaluation"]

    loaders = build_dataloaders(config)
    targets, predictions, confidences, images = predict(model, loaders[split], device)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=range(len(CLASS_NAMES)), zero_division=0
    )

    report: EvaluationReport = {
        "run_id": summary["run_id"],
        "model": summary["model"],
        "representation": summary["representation"],
        "split": split,
        "accuracy": (predictions == targets).float().mean().item(),
        "macro_f1": float(f1.mean()),
        "per_class_precision": dict(zip(CLASS_NAMES, precision.tolist())),
        "per_class_recall": dict(zip(CLASS_NAMES, recall.tolist())),
        "per_class_f1": dict(zip(CLASS_NAMES, f1.tolist())),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=range(len(CLASS_NAMES))).tolist(),
        "parameters": count_parameters(model),
        "training_seconds": summary["training_seconds"],
        "inference_ms_per_image": measure_latency(model, device, settings["inference"]),
    }

    shared_io.write_json(report_path(config, summary["run_id"], split), report)
    if settings["example_predictions"]:
        export_examples(
            images,
            targets,
            predictions,
            confidences,
            config["dataset"]["normalization"],
            settings["example_predictions"],
            evaluation_dir(config, summary["run_id"], split),
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score one run and write its evaluation report.")
    parser.add_argument("run", type=resolve_run, help="run id, or a path to the run directory")
    parser.add_argument("--split", choices=EVALUATION_SPLITS, default="val", help="test only after model selection")
    args = parser.parse_args(argv)

    report = evaluate(args.run, args.split)
    print(f"{report['model']:<12} {args.split:<5} accuracy {report['accuracy']:.4f}  macro-F1 {report['macro_f1']:.4f}")
    for batch_size, milliseconds in report["inference_ms_per_image"].items():
        print(f"{'':<12} latency at batch {batch_size:<4} {milliseconds:.4g} ms per image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
