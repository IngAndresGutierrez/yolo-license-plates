"""Split images (and their Pascal VOC annotations) into train/val/test.

Writes dataset/images/{split}/ and, when --annotations is given, copies each
image's original XML to dataset/annotations/{split}/ so the raw labels stay
available for other formats or experiments.
"""

import argparse
import random
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def split_names(
    names: list[str], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[str]]:
    """Shuffle deterministically and cut into train/val/test by ratio."""
    if abs(sum(ratios) - 1) > 1e-9:
        raise ValueError(f"Ratios must add up to 1, got {ratios}")

    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)

    n_train = round(len(shuffled) * ratios[0])
    n_val = round(len(shuffled) * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source", type=Path, help="Folder containing the images")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Folder with one <image-stem>.xml per image (Pascal VOC)",
    )
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--ratios", type=float, nargs=3, default=(0.7, 0.2, 0.1), metavar=SPLITS
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    images = [
        p.name for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise SystemExit(f"No images found in {args.source}")

    if args.annotations is not None:
        missing = [
            n
            for n in images
            if not (args.annotations / f"{Path(n).stem}.xml").is_file()
        ]
        if missing:
            raise SystemExit(
                f"{len(missing)} images have no annotation in {args.annotations}, "
                f"e.g. {missing[0]}"
            )

    for folder in ("images", "annotations"):
        existing = [p for s in SPLITS for p in (args.output / folder / s).glob("*")]
        if existing:
            raise SystemExit(
                f"{args.output / folder} already has {len(existing)} files; "
                "empty it first"
            )

    for split, names in split_names(images, tuple(args.ratios), args.seed).items():
        image_dir = args.output / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(args.source / name, image_dir / name)

        if args.annotations is not None:
            annotation_dir = args.output / "annotations" / split
            annotation_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                xml_name = f"{Path(name).stem}.xml"
                shutil.copy2(args.annotations / xml_name, annotation_dir / xml_name)

        print(f"{split:>5}: {len(names)} images")


if __name__ == "__main__":
    main()
