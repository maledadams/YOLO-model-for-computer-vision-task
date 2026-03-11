from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import requests


IMAGE_TIMEOUT_SECONDS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Ultralytics NDJSON export into a local YOLO dataset."
    )
    parser.add_argument("--input", required=True, help="Path to the .ndjson dataset export.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory where images/, labels/, and data.yaml will be created.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick experiments.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Keep images with no annotations by writing empty label files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download images and rewrite labels if files already exist.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_row: dict[str, Any] | None = None
    image_rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["type"] == "dataset":
                dataset_row = row
            elif row["type"] == "image":
                image_rows.append(row)

    if dataset_row is None:
        raise ValueError(f"No dataset metadata row was found in {path}.")

    return dataset_row, image_rows


def ensure_layout(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_label_file(label_path: Path, boxes: list[list[float]]) -> None:
    lines = []
    for class_id, x, y, w, h in boxes:
        lines.append(f"{int(class_id)} {x} {y} {w} {h}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def download_file(session: requests.Session, url: str, destination: Path, overwrite: bool) -> bool:
    if destination.exists() and not overwrite:
        return True

    response = session.get(url, timeout=IMAGE_TIMEOUT_SECONDS)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return True


def build_yaml(output_dir: Path, dataset_row: dict[str, Any]) -> None:
    names_map = dataset_row["class_names"]
    ordered_names = [names_map[str(index)] for index in range(len(names_map))]

    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for index, name in enumerate(ordered_names):
        lines.append(f"  {index}: {name}")
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    dataset_row, image_rows = load_rows(input_path)
    ensure_layout(output_dir)

    counts = Counter()
    failures: list[str] = []

    session = requests.Session()
    processed = 0

    for row in image_rows:
        if args.max_images is not None and processed >= args.max_images:
            break

        split = row.get("split", "train")
        boxes = row.get("annotations", {}).get("boxes", [])
        has_labels = bool(boxes)

        if not has_labels and not args.include_empty:
            counts["skipped_empty"] += 1
            continue

        image_path = output_dir / "images" / split / row["file"]
        label_path = output_dir / "labels" / split / f"{Path(row['file']).stem}.txt"

        try:
            download_file(session, row["url"], image_path, overwrite=args.overwrite)
            write_label_file(label_path, boxes)
            processed += 1
            counts[f"saved_{split}"] += 1
        except Exception as exc:  # pragma: no cover - best effort download path
            failures.append(f"{row['file']}: {exc}")
            counts["failed"] += 1

    build_yaml(output_dir, dataset_row)

    summary = {
        "dataset_name": dataset_row["name"],
        "source_ndjson": str(input_path),
        "output_dir": str(output_dir),
        "class_names": dataset_row["class_names"],
        "counts": dict(counts),
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
