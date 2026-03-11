from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


CLASS_NAMES = [
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a smaller YOLO dataset subset for quick local training.")
    parser.add_argument("--source", required=True, help="Source YOLO dataset directory.")
    parser.add_argument("--output", required=True, help="Output subset dataset directory.")
    parser.add_argument("--train", type=int, default=512, help="Number of training images.")
    parser.add_argument("--val", type=int, default=128, help="Number of validation images.")
    parser.add_argument("--test", type=int, default=16, help="Number of test images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def copy_split(source: Path, output: Path, split: str, limit: int, rng: random.Random) -> int:
    image_source = source / "images" / split
    label_source = source / "labels" / split
    image_output = output / "images" / split
    label_output = output / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_source.iterdir())
    if split == "test":
        selected = image_paths[: min(limit, len(image_paths))]
    else:
        selected = sorted(rng.sample(image_paths, min(limit, len(image_paths))))

    for image_path in selected:
        label_path = label_source / f"{image_path.stem}.txt"
        shutil.copy2(image_path, image_output / image_path.name)
        if label_path.exists():
            shutil.copy2(label_path, label_output / label_path.name)
        else:
            (label_output / label_path.name).write_text("", encoding="utf-8")

    return len(selected)


def write_yaml(output: Path) -> None:
    lines = [
        f"path: {output.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for index, name in enumerate(CLASS_NAMES):
        lines.append(f"  {index}: {name}")
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    rng = random.Random(args.seed)

    counts = {
        "train": copy_split(source, output, "train", args.train, rng),
        "val": copy_split(source, output, "val", args.val, rng),
        "test": copy_split(source, output, "test", args.test, rng),
    }
    write_yaml(output)

    print(f"Created subset dataset at {output}")
    for split, count in counts.items():
        print(f"{split}: {count}")


if __name__ == "__main__":
    main()
