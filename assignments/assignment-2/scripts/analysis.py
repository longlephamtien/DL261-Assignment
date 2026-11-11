"""Aggregates the evaluation reports into the numbers the report quotes.

    python scripts/analysis.py [--split test]

Prints the per-arm table, the decision-rule verdicts, the per-class comparison and the error
taxonomy. Nothing here is written to disk: src/figures.py owns the committed outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

import numpy as np

from src.interfaces import CLASS_NAMES

ARMS = ("e1-deeplabv3", "e1-unet", "e2-frozen", "e3-dice", "e4-resnet50")


def arm_of(summary: dict, config_text: str) -> str:
    if summary["model"] == "unet":
        return "e1-unet"
    if "resnet50" in summary["backbone"]:
        return "e4-resnet50"
    if re.search(r"^\s*loss:\s*dice", config_text, re.M):
        return "e3-dice"
    return "e2-frozen" if "freeze_backbone: true" in config_text else "e1-deeplabv3"


def load(split: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for summary_path in sorted(ROOT.glob("results/runs/*/summary.json")):
        directory = summary_path.parent
        report_path = ROOT / "results/evaluation" / directory.name / split / "report.json"
        if not report_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        history = json.loads((directory / "history.json").read_text())
        report = json.loads(report_path.read_text())
        report["seed"] = summary["seed"]
        report["epochs"] = summary["epochs_trained"]
        report["best_epoch"] = summary["best_epoch"]
        report["val_miou"] = summary["checkpoint_value"]
        report["seconds_per_epoch"] = st.mean(r["seconds"] for r in history)
        report["training_seconds"] = summary["training_seconds"]
        out.setdefault(arm_of(summary, (directory / "config.yaml").read_text()), []).append(report)
    return {arm: out[arm] for arm in ARMS if arm in out}


def interval(values: list[float]) -> tuple[float, float]:
    m = st.mean(values)
    s = st.stdev(values) if len(values) > 1 else 0.0
    return m, s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)
    arms = load(args.split)

    print(f"=== {args.split} split, {sum(len(v) for v in arms.values())} runs ===\n")
    head = (f"{'arm':<14}{'n':>2}{'params':>12}{'train':>8}{'s/ep':>7}{'ms@1':>7}{'ms@8':>7}"
            f"{'GB':>6}{'val mIoU':>19}{'mIoU':>19}{'Dice':>19}{'pixacc':>9}")
    print(head)
    stats = {}
    for arm, runs in arms.items():
        mi, si = interval([r["miou"] for r in runs])
        md, sd = interval([r["dice"] for r in runs])
        mv, sv = interval([r["val_miou"] for r in runs])
        stats[arm] = dict(miou=(mi, si), val=(mv, sv), dice=(md, sd), n=len(runs))
        print(f"{arm:<14}{len(runs):>2}{runs[0]['parameters']:>12,}"
              f"{st.mean(r['training_seconds'] for r in runs)/3600:>7.2f}h"
              f"{st.mean(r['seconds_per_epoch'] for r in runs):>7.1f}"
              f"{st.mean(r['inference_ms_per_image']['1'] for r in runs):>7.1f}"
              f"{st.mean(r['inference_ms_per_image']['8'] for r in runs):>7.1f}"
              f"{st.mean(r['peak_memory_bytes'] for r in runs)/1024**3:>6.2f}"
              f"{mv:>12.4f} +-{sv:<6.4f}{mi:>12.4f} +-{si:<6.4f}{md:>12.4f} +-{sd:<6.4f}"
              f"{st.mean(r['pixel_accuracy'] for r in runs):>9.4f}")

    print("\n=== decision rule on validation mIoU (mean +- 1 sd disjoint) ===")
    names = list(stats)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (ma, sa), (mb, sb) = stats[a]["val"], stats[b]["val"]
            if stats[a]["n"] < 2 or stats[b]["n"] < 2:
                verdict = "indicative only (one seed)"
            elif min(ma + sa, mb + sb) < max(ma - sa, mb - sb):
                hi, lo = (a, b) if ma > mb else (b, a)
                verdict = f"{hi} beats {lo} by {abs(ma-mb):.4f}"
            else:
                verdict = "indistinguishable"
            print(f"  {a:<14} vs {b:<14} -> {verdict}")

    print(f"\n=== per-class IoU on {args.split} (mean over seeds) ===")
    print(f"{'class':<15}" + "".join(f"{a.split('-')[-1]:>13}" for a in arms))
    per_class = {arm: {} for arm in arms}
    for name in CLASS_NAMES:
        row = f"{name:<15}"
        for arm, runs in arms.items():
            vals = [r["per_class_iou"][name] for r in runs if r["per_class_iou"].get(name) is not None]
            per_class[arm][name] = st.mean(vals) if vals else float("nan")
            row += f"{per_class[arm][name]:>13.4f}" if vals else f"{'--':>13}"
        print(row)

    print(f"\n=== error taxonomy, best arm on {args.split} ===")
    best = max(arms, key=lambda a: stats[a]["miou"][0])
    runs = arms[best]
    matrix = np.mean([np.array(r["confusion_matrix"], dtype=float) for r in runs], axis=0)
    total = matrix.sum()
    correct = np.trace(matrix)
    print(f"best arm {best}: {correct/total:.4%} of labelled pixels correct, "
          f"{(total-correct)/1e6:.1f}M of {total/1e6:.1f}M mislabelled")
    errors = matrix - np.diag(np.diag(matrix))
    flat = [(errors[t, p], CLASS_NAMES[t], CLASS_NAMES[p]) for t in range(19) for p in range(19) if t != p]
    flat.sort(reverse=True)
    print("  largest confusions (truth -> prediction, share of all error pixels):")
    for count, truth, pred in flat[:12]:
        print(f"    {truth:<14} -> {pred:<14} {count/1e6:>7.2f}M  {count/(total-correct):>6.1%}")
    print("  per-class recall and the class it is most often confused with:")
    for t, name in enumerate(CLASS_NAMES):
        support = matrix[t].sum()
        if support == 0:
            continue
        worst = max(range(19), key=lambda p: errors[t, p])
        print(f"    {name:<14} support {support/1e6:>7.2f}M  recall {matrix[t,t]/support:>6.1%}"
              f"  IoU {per_class[best][name]:>6.4f}  -> {CLASS_NAMES[worst]:<14} {errors[t,worst]/support:>5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
