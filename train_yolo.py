from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model on the chess dataset.")
    parser.add_argument("--data", required=True, help="Path to data.yaml.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model checkpoint.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default=None, help="CUDA device id, 'cpu', or omitted.")
    parser.add_argument("--project", default="runs/detect", help="Ultralytics project directory.")
    parser.add_argument("--name", default="chess-pieces", help="Run name.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    project_path = Path(args.project).resolve()

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_path),
        name=args.name,
        workers=args.workers,
        patience=args.patience,
    )
    model.val(data=str(data_path))


if __name__ == "__main__":
    main()
