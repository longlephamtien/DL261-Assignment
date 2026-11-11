"""How many of the 19 classes a training crop and a training batch actually contain.

    python scripts/class_presence.py [--batches 12]

The Dice loss averages over all 19 classes whether or not the batch contains them, so this
measures how much of that objective is contributed by classes with no positive pixel.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from shared import io as shared_io
from src.data import build_dataloaders
from src.interfaces import IGNORE_INDEX, NUM_CLASSES


def merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=12)
    args = parser.parse_args()

    config = merge(shared_io.read_yaml(ROOT / "configs/base.yaml"),
                   shared_io.read_yaml(ROOT / "configs/deeplabv3.yaml"))
    config["seed"] = 0
    config["loader"]["num_workers"] = 0  # the measurement is short; workers cost more than they save

    loaders = build_dataloaders(config)
    iterator = iter(loaders["train"])
    per_crop: list[int] = []
    per_batch: list[int] = []
    for _ in range(args.batches):
        _, targets = next(iterator)
        for index in range(targets.shape[0]):
            unique = targets[index].unique()
            per_crop.append(int((unique != IGNORE_INDEX).sum()))
        unique = targets.unique()
        per_batch.append(int((unique != IGNORE_INDEX).sum()))

    print(f"{len(per_crop)} crops in {len(per_batch)} batches of {config['loader']['batch_size']}")
    print(f"classes per 512 crop  : mean {st.mean(per_crop):.2f} of {NUM_CLASSES} "
          f"(min {min(per_crop)}, max {max(per_crop)})")
    print(f"classes per batch     : mean {st.mean(per_batch):.2f} of {NUM_CLASSES} "
          f"(min {min(per_batch)}, max {max(per_batch)})")
    print(f"Dice terms with no positive pixel in the batch: "
          f"{1 - st.mean(per_batch) / NUM_CLASSES:.1%} on average")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
