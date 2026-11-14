"""RefCOCO download, verification, and the pair statistics the proposal reports.

    python -m src.data --config configs/base.yaml [--annotations | --stats]
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
import statistics as st
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from shared import io as shared_io

from .interfaces import PAD_TOKEN, SPLITS, Batch, Pair, Ref
from .utils import load_config, resolve_path

MIRRORED = {"left": "right", "right": "left", "leftmost": "rightmost", "rightmost": "leftmost"}
"""Words a horizontal flip inverts. 46.3% of expressions contain one, so a flip that does not
rewrite them relabels nearly half the data."""

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
"""The statistics the pretrained vision weights were fitted under."""

ANNOTATION_FILES = ("instances.json", "refs(unc).p")
"""What the archive must contain."""


RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
"""Retryable: the Internet Archive rate-limits, so 429 is expected under load."""


def fetch(url: str, destination: Path, description: str, attempts: int = 5) -> Path:
    """Download unless already present, retrying with a growing backoff."""
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                if response.status_code in RETRY_STATUS:
                    raise requests.HTTPError(f"{response.status_code} from {url}", response=response)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with partial.open("wb") as handle, tqdm(
                    total=total or None, unit="B", unit_scale=True, desc=description
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        handle.write(chunk)
                        bar.update(len(chunk))
            partial.rename(destination)
            return destination
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as error:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            delay = 2 ** attempt
            print(f"  {description}: {error}; retrying in {delay}s ({attempt}/{attempts - 1})")
            time.sleep(delay)
    return destination


def prepare_annotations(config: dict) -> Path:
    """Fetch and unpack the annotations, refusing a copy whose checksum does not match."""
    dataset = config["dataset"]
    root = resolve_path(dataset["root"])
    archive = root / "refcoco.zip"
    folder = root / "refcoco"

    if all((folder / name).is_file() for name in ANNOTATION_FILES):
        return folder

    fetch(dataset["annotations"], archive, "refcoco.zip")
    expected = dataset["annotations_sha256"]
    actual = shared_io.sha256(archive)
    if actual != expected:
        raise ValueError(
            f"{archive} has checksum {actual}, not the committed {expected}; "
            "the mirror has changed and must not be used until the difference is explained"
        )

    with zipfile.ZipFile(archive) as unpacked:
        unpacked.extractall(root)
    missing = [name for name in ANNOTATION_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{archive} unpacked without {', '.join(missing)}")
    return folder


def load_refs(folder: Path) -> tuple[list[Ref], dict]:
    """The referred objects of the UNC partition, and the COCO instances they point into."""
    refs = pickle.load((folder / "refs(unc).p").open("rb"))
    instances = json.loads((folder / "instances.json").read_text())
    return refs, instances


def _fetch_image(url: str, target: Path, session: requests.Session, attempts: int = 3) -> None:
    """One image, written through a temporary name so an interruption leaves no half file."""
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=60)
            if response.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"{response.status_code} from {url}", response=response)
            response.raise_for_status()
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_bytes(response.content)
            partial.rename(target)
            return
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)


def prepare_images(config: dict, refs: list[Ref], instances: dict, workers: int = 16) -> Path:
    """Fetch only the referenced images, several at a time."""
    dataset = config["dataset"]
    images = resolve_path(dataset["root"]) / "images"
    images.mkdir(parents=True, exist_ok=True)
    names = {i["id"]: i["file_name"] for i in instances["images"]}
    wanted = sorted({names[ref["image_id"]] for ref in refs})
    pending = [name for name in wanted if not (images / name).is_file()]
    print(f"{len(wanted):,} images referenced, {len(pending):,} still to fetch")

    local = threading.local()

    def session() -> requests.Session:
        """One session per worker, keeping connections alive."""
        if not hasattr(local, "session"):
            local.session = requests.Session()
        return local.session

    failed: list[str] = []

    def download(name: str) -> None:
        try:
            _fetch_image(f"{dataset['images']}/{name}", images / name, session())
        except Exception:  # noqa: BLE001  one bad image must not end the run
            failed.append(name)

    with ThreadPoolExecutor(workers) as pool:
        list(tqdm(pool.map(download, pending), total=len(pending), desc="images", unit="img"))

    if failed:
        print(f"{len(failed)} images failed; re-running fetches only those")
    return images


def build_split(refs: list[Ref]) -> dict:
    """Materialise the UNC partition as image ids, the unit a split must respect."""
    images = {split: sorted({r["image_id"] for r in refs if r["split"] == split}) for split in SPLITS}
    return {
        "dataset": "refcoco",
        "partition": "unc",
        "split_unit": "image",
        "images": images,
        "counts": {
            split: {
                "images": len(images[split]),
                "objects": sum(1 for r in refs if r["split"] == split),
                "expressions": sum(len(r["sentences"]) for r in refs if r["split"] == split),
            }
            for split in SPLITS
        },
    }


def verify_split(split: dict, refs: list[Ref]) -> None:
    """Fail if any image appears twice or any ref falls outside its split's image list."""
    seen: dict[int, str] = {}
    for name, ids in split["images"].items():
        for image_id in ids:
            if image_id in seen:
                raise ValueError(f"image {image_id} is in both {seen[image_id]} and {name}")
            seen[image_id] = name
    stray = [r["ref_id"] for r in refs if seen.get(r["image_id"]) != r["split"]]
    if stray:
        raise ValueError(f"{len(stray)} refs sit outside their split's image list, first {stray[0]}")


def distribution(refs: list[Ref], instances: dict) -> dict:
    """Every pair-quality and distribution figure the proposal quotes."""
    annotations = {a["id"]: a for a in instances["annotations"]}
    images = {i["id"]: i for i in instances["images"]}
    categories = {c["id"]: c["name"] for c in instances["categories"]}
    per_image_category = collections.Counter(
        (a["image_id"], a["category_id"]) for a in instances["annotations"]
    )

    expressions = [sentence for ref in refs for sentence in ref["sentences"]]
    lengths = sorted(len(sentence["tokens"]) for sentence in expressions)
    tokens = [token for sentence in expressions for token in sentence["tokens"]]
    distractors = [per_image_category[(r["image_id"], r["category_id"])] for r in refs]
    areas = sorted(
        annotations[r["ann_id"]]["area"] / (images[r["image_id"]]["width"] * images[r["image_id"]]["height"])
        for r in refs
    )

    def quantile(values: list[float], fraction: float) -> float:
        return values[min(int(fraction * len(values)), len(values) - 1)]

    by_split = {}
    for split in SPLITS:
        members = [r for r in refs if r["split"] == split]
        by_split[split] = {
            "objects": len(members),
            "images": len({r["image_id"] for r in members}),
            "expressions": sum(len(r["sentences"]) for r in members),
        }

    # Measured, not taken on trust.
    image_sets = {split: {r["image_id"] for r in refs if r["split"] == split} for split in SPLITS}
    overlap = {
        f"{a}|{b}": len(image_sets[a] & image_sets[b])
        for index, a in enumerate(SPLITS)
        for b in SPLITS[index + 1:]
    }

    return {
        "objects": len(refs),
        "images": len({r["image_id"] for r in refs}),
        "expressions": len(expressions),
        "expressions_per_object": len(expressions) / len(refs),
        "missing": {
            "expression": sum(1 for r in refs if not r["sentences"]),
            "image": sum(1 for r in refs if r["image_id"] not in images),
            "mask": sum(1 for r in refs if r["ann_id"] not in annotations),
        },
        "split": by_split,
        "split_image_overlap": overlap,
        "expression_tokens": {
            "mean": st.mean(lengths),
            "median": quantile(lengths, 0.50),
            "p90": quantile(lengths, 0.90),
            "min": lengths[0],
            "max": lengths[-1],
            "share_at_most_two": sum(1 for n in lengths if n <= 2) / len(lengths),
            "vocabulary": len(set(tokens)),
            "tokens": len(tokens),
        },
        "distractors": {
            "mean": st.mean(distractors),
            "median": quantile(sorted(distractors), 0.50),
            "max": max(distractors),
            "histogram": {str(k): v for k, v in sorted(collections.Counter(distractors).items())},
            "sole_instance": sum(1 for n in distractors if n == 1) / len(distractors),
        },
        "area_share": {
            "p10": quantile(areas, 0.10),
            "p25": quantile(areas, 0.25),
            "median": quantile(areas, 0.50),
            "p75": quantile(areas, 0.75),
            "p90": quantile(areas, 0.90),
            "under_one_percent": sum(1 for a in areas if a < 0.01) / len(areas),
        },
        "categories": {
            "distinct": len({r["category_id"] for r in refs}),
            "top": collections.Counter(categories[r["category_id"]] for r in refs).most_common(8),
        },
        "licenses": _licenses(instances, refs),
        "resolutions": len({(images[r["image_id"]]["width"], images[r["image_id"]]["height"]) for r in refs}),
    }


def _licenses(instances: dict, refs: list[Ref]) -> dict:
    """Images per declared licence; COCO licenses each image separately."""
    names = {licence["id"]: licence["name"] for licence in instances["licenses"]}
    referenced = {r["image_id"] for r in refs}
    counts = collections.Counter(
        i.get("license") for i in instances["images"] if i["id"] in referenced
    )
    return {names.get(key, str(key)): value for key, value in counts.most_common()}


def build_pairs(refs: list[Ref], instances: dict, split: str, per_object: int | None = None) -> list[Pair]:
    """Flatten refs into the prediction unit, one row per expression.

    `per_object` applies the training subset rule: keep at most that many expressions of each
    object. Evaluation passes None so no reported number depends on the rule.
    """
    names = {i["id"]: i["file_name"] for i in instances["images"]}
    pairs: list[Pair] = []
    for ref in refs:
        if ref["split"] != split:
            continue
        sentences = ref["sentences"] if per_object is None else ref["sentences"][:per_object]
        for sentence in sentences:
            pairs.append({
                "ref_id": ref["ref_id"],
                "ann_id": ref["ann_id"],
                "image_id": ref["image_id"],
                "file_name": names[ref["image_id"]],
                "sentence": sentence["sent"],
                "tokens": sentence["tokens"],
            })
    return pairs


def build_vocabulary(refs: list[Ref], minimum: int = 2) -> dict[str, int]:
    """Token to id, built from the training split alone so no test word leaks into the model."""
    counts = collections.Counter(
        token
        for ref in refs if ref["split"] == "train"
        for sentence in ref["sentences"]
        for token in sentence["tokens"]
    )
    vocabulary = {"<pad>": PAD_TOKEN, "<unk>": 1}
    for token, count in counts.most_common():
        if count >= minimum:
            vocabulary[token] = len(vocabulary)
    return vocabulary


class RefCOCOPairs(Dataset):
    """One expression, the image holding its object, and that object's mask.

    The mask is rasterised from the COCO polygon on demand; `instances.json` stores no raster.
    """

    def __init__(self, pairs: list[Pair], instances: dict, images: Path,
                 vocabulary: dict[str, int], size: tuple[int, int], augment: bool = False,
                 flip_probability: float = 0.5) -> None:
        self.pairs = pairs
        self.annotations = {a["id"]: a for a in instances["annotations"]}
        self.shapes = {i["id"]: (i["height"], i["width"]) for i in instances["images"]}
        self.images = images
        self.vocabulary = vocabulary
        self.size = size
        self.augment = augment
        self.flip_probability = flip_probability

    def __len__(self) -> int:
        return len(self.pairs)

    def encode(self, tokens: list[str]) -> list[int]:
        """Token ids, with anything below the vocabulary threshold mapped to <unk>."""
        return [self.vocabulary.get(token, 1) for token in tokens]

    def mask(self, pair: Pair) -> np.ndarray:
        """Rasterise the referred object's polygon at the image's own resolution."""
        height, width = self.shapes[pair["image_id"]]
        rles = mask_utils.frPyObjects(self.annotations[pair["ann_id"]]["segmentation"], height, width)
        return mask_utils.decode(mask_utils.merge(rles))

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        image = Image.open(self.images / pair["file_name"]).convert("RGB")
        mask = Image.fromarray(self.mask(pair) * 255)
        tokens = pair["tokens"]

        image = image.resize(self.size[::-1], Image.BILINEAR)
        # Nearest neighbour only: interpolating a binary mask invents values in neither class.
        mask = mask.resize(self.size[::-1], Image.NEAREST)

        if self.augment and random.random() < self.flip_probability:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            tokens = [MIRRORED.get(token, token) for token in tokens]

        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div_(255)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": tensor,
            "tokens": torch.tensor(self.encode(tokens), dtype=torch.long),
            "mask": torch.from_numpy(np.array(mask) > 127).long(),
            "ref_id": pair["ref_id"],
        }


def collate(samples: list[dict]) -> Batch:
    """Pad the token sequences to the longest in the batch and mark the real positions."""
    longest = max(len(s["tokens"]) for s in samples)
    tokens = torch.full((len(samples), longest), PAD_TOKEN, dtype=torch.long)
    attention = torch.zeros((len(samples), longest), dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = len(sample["tokens"])
        tokens[row, :length] = sample["tokens"]
        attention[row, :length] = True
    return {
        "images": torch.stack([s["image"] for s in samples]),
        "tokens": tokens,
        "attention": attention,
        "masks": torch.stack([s["mask"] for s in samples]),
        "ref_ids": torch.tensor([s["ref_id"] for s in samples], dtype=torch.long),
    }


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """One loader per split, sharing the training vocabulary and the committed partition."""
    folder = prepare_annotations(config)
    refs, instances = load_refs(folder)
    images = resolve_path(config["dataset"]["root"]) / "images"
    vocabulary = build_vocabulary(refs)
    loader = config["loader"]
    size = tuple(config["input"]["size"])

    loaders = {}
    for split in SPLITS:
        training = split == "train"
        pairs = build_pairs(refs, instances, split,
                            loader["expressions_per_object"] if training else None)
        dataset = RefCOCOPairs(pairs, instances, images, vocabulary, size,
                               augment=training and loader["augment"],
                               flip_probability=loader["flip_probability"])
        loaders[split] = DataLoader(
            dataset, batch_size=loader["batch_size"], shuffle=training,
            num_workers=loader["num_workers"], collate_fn=collate, drop_last=training,
        )
    return loaders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download RefCOCO and measure its pair statistics.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--annotations", action="store_true", help="fetch the annotations and stop")
    parser.add_argument("--stats", action="store_true", help="write results/eda/distribution.json and stop")
    parser.add_argument("--workers", type=int, default=16, help="parallel image downloads")
    parser.add_argument("--split", action="store_true", help="write splits/refcoco-unc.json and stop")
    args = parser.parse_args(argv)

    config = load_config(resolve_path(args.config))
    folder = prepare_annotations(config)
    print(folder)
    if args.annotations:
        return 0

    refs, instances = load_refs(folder)
    if args.split:
        split = build_split(refs)
        verify_split(split, refs)
        shared_io.write_json(resolve_path("splits/refcoco-unc.json"), split)
        for name, counts in split["counts"].items():
            print(f"{name:<6} {counts['images']:>6,} images  {counts['objects']:>6,} objects  "
                  f"{counts['expressions']:>7,} expressions")
        return 0
    if args.stats:
        measured = distribution(refs, instances)
        shared_io.write_json(resolve_path("results/eda/distribution.json"), measured)
        print(f"{measured['expressions']:,} expressions over {measured['objects']:,} objects "
              f"in {measured['images']:,} images")
        print(f"missing modalities: {measured['missing']}")
        print(f"split image overlap: {measured['split_image_overlap']}")
        return 0

    prepare_images(config, refs, instances, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
