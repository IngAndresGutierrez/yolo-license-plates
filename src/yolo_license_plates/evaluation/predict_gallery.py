"""Run inference on the test split, save predictions and build a gallery.

For every test image this writes:

- predictions/<stem>.txt   YOLO format with confidence: class xc yc w h conf
- gallery/<status>/<stem>.jpg  original image (left, no marks) next to the
  prediction (right) with boxes and confidence

Predictions are matched to the ground-truth labels (IoU >= --iou) to classify
each image as correct, with false positives, with missed plates, or without
detections. predictions.json, summary.md and index.html summarize the result.

    python -m yolo_license_plates.evaluation.predict_gallery --weights models/best.pt
"""

import argparse
import html
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Ordered from best to worst; also the gallery subfolders
STATUSES = {
    "correct": "Correcta: todas las placas detectadas, sin falsos positivos",
    "false_positive": "Con falsos positivos: detecta algo que no es una placa",
    "missed": "Con placas no detectadas (falsos negativos)",
    "mixed": "Con falsos positivos y placas no detectadas",
    "no_detections": "Sin detecciones",
}

TP_COLOR = (0, 200, 0)
FP_COLOR = (230, 0, 0)
MISSED_COLOR = (255, 200, 0)


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float | None = None


@dataclass
class ImageResult:
    name: str
    predictions: list[Box]
    ground_truth: list[Box]
    true_positives: list[int] = field(default_factory=list)  # prediction indexes
    false_positives: list[int] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)  # ground-truth indexes
    status: str = ""


def iou(a: Box, b: Box) -> float:
    inter_w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    inter_h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = inter_w * inter_h
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def match(
    predictions: list[Box], ground_truth: list[Box], iou_threshold: float
) -> tuple[list[int], list[int], list[int]]:
    """Greedy matching by descending confidence, one ground truth per prediction.

    Returns (true-positive prediction indexes, false-positive prediction
    indexes, missed ground-truth indexes).
    """
    order = sorted(range(len(predictions)), key=lambda i: -(predictions[i].conf or 0))
    unmatched_gt = set(range(len(ground_truth)))
    true_positives, false_positives = [], []
    for i in order:
        best_j, best_iou = None, iou_threshold
        for j in unmatched_gt:
            overlap = iou(predictions[i], ground_truth[j])
            if overlap >= best_iou:
                best_j, best_iou = j, overlap
        if best_j is None:
            false_positives.append(i)
        else:
            true_positives.append(i)
            unmatched_gt.remove(best_j)
    return sorted(true_positives), sorted(false_positives), sorted(unmatched_gt)


def classify(result: ImageResult) -> str:
    if not result.predictions:
        return "no_detections"
    if result.false_positives and result.missed:
        return "mixed"
    if result.false_positives:
        return "false_positive"
    if result.missed:
        return "missed"
    return "correct"


def read_yolo_labels(path: Path, width: int, height: int) -> list[Box]:
    """Ground-truth YOLO labels (normalized xywh) as pixel boxes."""
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        _, xc, yc, w, h = map(float, line.split()[:5])
        boxes.append(
            Box(
                (xc - w / 2) * width,
                (yc - h / 2) * height,
                (xc + w / 2) * width,
                (yc + h / 2) * height,
            )
        )
    return boxes


def to_yolo_line(box: Box, width: int, height: int, class_id: int = 0) -> str:
    xc = (box.x1 + box.x2) / 2 / width
    yc = (box.y1 + box.y2) / 2 / height
    w = (box.x2 - box.x1) / width
    h = (box.y2 - box.y1) / height
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {box.conf:.4f}"


STATUS_LABELS = {
    "correct": "correcta",
    "false_positive": "con falsos positivos",
    "missed": "con placas no detectadas",
    "mixed": "con falsos positivos y placas no detectadas",
    "no_detections": "sin detecciones",
}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Unicode TTF (PIL's built-in font has no accents); falls back to the default."""
    try:
        from ultralytics.utils.checks import check_font

        return ImageFont.truetype(str(check_font("Arial.Unicode.ttf")), size)
    except (ImportError, OSError):  # no ultralytics, or font download failed
        return ImageFont.load_default(size=size)


def render_side_by_side(image: Image.Image, result: ImageResult) -> Image.Image:
    """Original on the left, prediction with boxes and confidence on the right.

    A two-line header gives the panel titles and, across the full width, the
    image name and its outcome. True-positive labels go above their box and
    false-positive labels below it, so duplicate detections stay readable.
    """
    width, height = image.size
    font_size = max(13, height // 24)
    font = load_font(font_size)
    line_width = max(2, height // 150)
    gap = 10
    line_h = font_size + 8
    header = 2 * line_h + 6
    x0 = width + gap

    canvas = Image.new("RGB", (width * 2 + gap, height + header), "white")
    canvas.paste(image, (0, header))
    canvas.paste(image, (x0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "Original", fill="black", font=font)
    draw.text(
        (x0 + 4, 4),
        f"Predicción: {len(result.predictions)} cajas",
        fill="black",
        font=font,
    )
    summary = (
        f"{result.name} · {STATUS_LABELS[result.status]} · "
        f"{len(result.true_positives)} TP, {len(result.false_positives)} FP, "
        f"{len(result.missed)} FN"
    )
    # Shrink the summary line until it fits narrow images
    summary_font, size = font, font_size
    while (
        draw.textbbox((0, 0), summary, font=summary_font)[2] > canvas.width - 8
        and size > 8
    ):
        size -= 1
        summary_font = load_font(size)
    draw.text((4, 4 + line_h), summary, fill=(80, 80, 80), font=summary_font)

    def draw_box(box: Box, color: tuple, label: str, below: bool = False) -> None:
        rect = (x0 + box.x1, header + box.y1, x0 + box.x2, header + box.y2)
        draw.rectangle(rect, outline=color, width=line_width)
        text_h = draw.textbbox((0, 0), label, font=font)[3]
        above_top = rect[1] - text_h - 4
        below_top = rect[3] + 2
        if below:
            top = below_top if below_top + text_h <= header + height else above_top
        else:
            top = above_top if above_top >= header else below_top
        # Keep the label inside the right panel
        text_w = draw.textbbox((0, 0), label, font=font)[2]
        left = min(rect[0], x0 + width - text_w - 2)
        draw.rectangle(draw.textbbox((left, top), label, font=font), fill=color)
        text_color = "black" if color == MISSED_COLOR else "white"
        draw.text((left, top), label, fill=text_color, font=font)

    for j in result.missed:
        draw_box(result.ground_truth[j], MISSED_COLOR, "no detectada")
    for i, box in enumerate(result.predictions):
        is_hit = i in result.true_positives
        draw_box(
            box,
            TP_COLOR if is_hit else FP_COLOR,
            f"licence {box.conf:.2f}",
            below=not is_hit,
        )
    return canvas


def render_summary(results: list[ImageResult], meta: dict) -> str:
    counts = {s: sum(r.status == s for r in results) for s in STATUSES}
    tp = sum(len(r.true_positives) for r in results)
    fp = sum(len(r.false_positives) for r in results)
    fn = sum(len(r.missed) for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    lines = [
        "# Inferencia sobre test",
        "",
        f"- Pesos: `{meta['weights']}`",
        f"- Imágenes: {len(results)} (`{meta['images']}`)",
        (
            f"- `imgsz` = {meta['imgsz']}, confianza mínima = {meta['conf']}, "
            f"IoU para considerar acierto = {meta['iou']}"
        ),
        "",
        "| Resultado por imagen | Imágenes |",
        "|---|---|",
        *(f"| {STATUSES[s]} | {counts[s]} |" for s in STATUSES),
        "",
        "| Cajas | |",
        "|---|---|",
        f"| Verdaderos positivos | {tp} |",
        f"| Falsos positivos | {fp} |",
        f"| Placas no detectadas | {fn} |",
        f"| Precision | {precision:.3f} |",
        f"| Recall | {recall:.3f} |",
        "",
        (
            "Colores en la galería: verde = acierto, rojo = falso positivo, "
            "amarillo = placa real no detectada (dibujada desde la etiqueta)."
        ),
        "",
    ]
    for status in STATUSES:
        names = [r.name for r in results if r.status == status]
        if names:
            lines.append(f"- **{status}**: {', '.join(names)}")
    return "\n".join(lines) + "\n"


def render_index(results: list[ImageResult]) -> str:
    sections = []
    for status, title in STATUSES.items():
        items = [r for r in results if r.status == status]
        if not items:
            continue
        figures = "\n".join(
            f'<figure><img src="gallery/{status}/{html.escape(Path(r.name).stem)}.jpg" '
            f'loading="lazy"><figcaption>{html.escape(r.name)} · '
            f"{len(r.true_positives)} TP, {len(r.false_positives)} FP, "
            f"{len(r.missed)} no detectadas</figcaption></figure>"
            for r in items
        )
        sections.append(f"<h2>{html.escape(title)} ({len(items)})</h2>\n{figures}")
    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Inferencia sobre test</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
figure {{ margin: 0 0 1.5rem; }} img {{ max-width: 100%; border: 1px solid #ccc; }}
figcaption {{ color: #555; font-size: .9rem; }}
</style></head><body>
<h1>Inferencia sobre test</h1>
<p>Izquierda: imagen original. Derecha: predicción con confianza.
Verde = acierto, rojo = falso positivo, amarillo = placa no detectada.</p>
{body}
</body></html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict on test images and build a side-by-side gallery."
    )
    parser.add_argument("--weights", type=Path, required=True, help="Trained .pt file")
    parser.add_argument("--images", type=Path, default=Path("dataset/images/test"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels/test"))
    parser.add_argument("--output", type=Path, default=Path("reports/test_predictions"))
    parser.add_argument(
        "--imgsz", type=int, help="Default: the imgsz the weights were trained with"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU to count a hit")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    images = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    imgsz = args.imgsz or model.ckpt.get("train_args", {}).get("imgsz", 640)

    predictions_dir = args.output / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for status in STATUSES:
        (args.output / "gallery" / status).mkdir(parents=True, exist_ok=True)
        for old in (args.output / "gallery" / status).glob("*.jpg"):
            old.unlink()

    results = []
    for path in images:
        prediction = model.predict(
            path, imgsz=imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        boxes = [
            Box(*xyxy, conf=float(conf))
            for xyxy, conf in zip(
                prediction.boxes.xyxy.tolist(),
                prediction.boxes.conf.tolist(),
                strict=True,
            )
        ]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        ground_truth = read_yolo_labels(args.labels / f"{path.stem}.txt", width, height)

        result = ImageResult(path.name, boxes, ground_truth)
        result.true_positives, result.false_positives, result.missed = match(
            boxes, ground_truth, args.iou
        )
        result.status = classify(result)
        results.append(result)

        (predictions_dir / f"{path.stem}.txt").write_text(
            "".join(to_yolo_line(b, width, height) + "\n" for b in boxes)
        )
        render_side_by_side(image, result).save(
            args.output / "gallery" / result.status / f"{path.stem}.jpg", quality=90
        )

    meta = {
        "weights": str(args.weights),
        "images": str(args.images),
        "imgsz": imgsz,
        "conf": args.conf,
        "iou": args.iou,
    }
    (args.output / "predictions.json").write_text(
        json.dumps(
            {
                **meta,
                "images_results": [
                    {
                        "image": r.name,
                        "status": r.status,
                        "predictions": [
                            {
                                "xyxy": [round(v, 1) for v in (b.x1, b.y1, b.x2, b.y2)],
                                "conf": round(b.conf, 4),
                                "match": "tp" if i in r.true_positives else "fp",
                            }
                            for i, b in enumerate(r.predictions)
                        ],
                        "missed_ground_truth": [
                            [round(v, 1) for v in (g.x1, g.y1, g.x2, g.y2)]
                            for j, g in enumerate(r.ground_truth)
                            if j in r.missed
                        ],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
        + "\n"
    )
    summary = render_summary(results, meta)
    (args.output / "summary.md").write_text(summary)
    (args.output / "index.html").write_text(render_index(results))
    print(summary)


if __name__ == "__main__":
    main()
