"""Train a YOLO license-plate detector with Ultralytics."""

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO license-plate detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/data.yaml"),
        help="Ultralytics dataset YAML",
    )
    parser.add_argument(
        "--model", default="yolo11n.pt", help="Base weights or model YAML"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device", default=None, help="cpu, 0, 0,1... (None = auto-detect)"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--project", type=Path, default=Path("runs"), help="Output directory for runs"
    )
    parser.add_argument("--name", default="train", help="Run name inside --project")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f"Dataset config not found: {args.data}")

    # Imported lazily so --help doesn't pay torch's import cost
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
