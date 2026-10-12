"""Convert Pascal VOC annotations to YOLO bounding-box labels.

Reads dataset/annotations/{split}/*.xml and writes one dataset/labels/{split}/
<image-stem>.txt per image, with a line per box:

    <class-id> <x-center> <y-center> <width> <height>

Coordinates are normalized to [0, 1] by the image size stored in the XML.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

SPLITS = ("train", "val", "test")
CLASSES = {"licence": 0}


def voc_box_to_yolo(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert (xmin, ymin, xmax, ymax) in pixels to normalized YOLO xywh."""
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(0.0, xmin), min(float(width), xmax)
    ymin, ymax = max(0.0, ymin), min(float(height), ymax)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Empty box {box} for image {width}x{height}")

    return (
        (xmin + xmax) / 2 / width,
        (ymin + ymax) / 2 / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def convert_annotation(xml_path: Path) -> list[str]:
    """Return the YOLO label lines for one VOC XML file."""
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))

    lines = []
    for obj in root.iter("object"):
        name = obj.findtext("name")
        if name not in CLASSES:
            raise ValueError(f"{xml_path}: unknown class {name!r}")
        box = tuple(
            float(obj.findtext(f"bndbox/{key}"))
            for key in ("xmin", "ymin", "xmax", "ymax")
        )
        x, y, w, h = voc_box_to_yolo(box, width, height)
        lines.append(f"{CLASSES[name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotations to YOLO labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    for split in SPLITS:
        annotation_dir = args.dataset / "annotations" / split
        label_dir = args.dataset / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)

        xml_paths = sorted(annotation_dir.glob("*.xml"))
        if not xml_paths:
            raise SystemExit(f"No annotations found in {annotation_dir}")

        n_boxes = 0
        for xml_path in xml_paths:
            lines = convert_annotation(xml_path)
            (label_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines) + "\n")
            n_boxes += len(lines)
        print(f"{split:>5}: {len(xml_paths)} labels, {n_boxes} boxes")


if __name__ == "__main__":
    main()
