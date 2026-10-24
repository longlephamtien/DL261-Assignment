"""Metrics, latency, and prediction overlays for one finished run.

    python -m src.evaluate <run_id> [--split val]

Validation is the default because Cityscapes withholds the test labels. Output goes to
`results/evaluation/<run_id>/<split>/` and is published with the run. Implemented by #63.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from shared import io as shared_io
from shared.device import count_parameters, select_device, synchronize

from .data import build_dataloaders
from .interfaces import CLASS_NAMES, EVALUATION_SPLITS, EvaluationReport, RunSummary, Split
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


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, keep: int = 0) -> tuple[ConfusionMatrix, dict[str, Tensor]]:
    """Accumulate the confusion matrix over a split, keeping a few batches for overlays.

    Masks are never concatenated across the split: at 512 by 512 a Cityscapes validation pass
    would hold hundreds of millions of pixels, so only the running matrix survives each batch.
    """
    model.eval().to(device)
    matrix = ConfusionMatrix()
    examples: dict[str, list[Tensor]] = {"images": [], "targets": [], "predictions": []}
    kept = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predictions = model(images).argmax(dim=1)
        matrix.update(targets.cpu(), predictions.cpu())

        if kept < keep:
            take = min(keep - kept, images.shape[0])
            examples["images"].append(images[:take].cpu())
            examples["targets"].append(targets[:take].cpu())
            examples["predictions"].append(predictions[:take].cpu())
            kept += take

    return matrix, {name: torch.cat(parts) for name, parts in examples.items() if parts}


def export_overlays(images: Tensor, targets: Tensor, predictions: Tensor, count: int, directory: Path) -> None:
    """Predicted masks drawn over their images, best and worst by image mIoU. Implemented by #74."""
    raise NotImplementedError("#74 implements the qualitative evaluation")


def measure_latency(model: nn.Module, device: torch.device, crop: list[int], inference: dict) -> dict[str, float]:
    """Milliseconds per image at each configured batch size, after warm-up, on one device."""
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

    loaders = build_dataloaders(config)
    matrix, examples = predict(model, loaders[split], device, settings["example_predictions"])
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
    }

    shared_io.write_json(report_path(config, summary["run_id"], split), report)
    del examples
    return report


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
