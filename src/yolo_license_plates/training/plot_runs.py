"""Plot validation mAP per epoch for several runs, one panel per metric.

The dot marks the epoch of best.pt (Ultralytics keeps the best mAP50-95), in
both panels, so the mAP50 panel shows that checkpoint's mAP50.

python -m yolo_license_plates.training.plot_runs runs/a_imgsz416 runs/b_imgsz640
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_license_plates.training.train import read_epoch_log

SELECTION_KEY = "metrics/mAP50-95(B)"
# Categorical slots 1-2 of the reference palette (validated for CVD on light)
COLORS = ("#2a78d6", "#eb6834")
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
METRICS = {"metrics/mAP50-95(B)": "mAP50-95", "metrics/mAP50(B)": "mAP50"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/figures/val_map_por_epoca.png")
    )
    args = parser.parse_args(argv)
    if len(args.runs) > len(COLORS):
        raise SystemExit(f"At most {len(COLORS)} runs")

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED})
    fig, axes = plt.subplots(
        1, len(METRICS), figsize=(10, 3.6), sharex=True, facecolor=SURFACE
    )
    for ax, (key, label) in zip(axes, METRICS.items(), strict=True):
        ax.set_facecolor(SURFACE)
        for run, color in zip(args.runs, COLORS, strict=False):
            rows = read_epoch_log(run / "results.csv")
            epochs = [r["epoch"] for r in rows]
            values = [r[key] for r in rows]
            best = max(range(len(rows)), key=lambda i: rows[i][SELECTION_KEY])
            # The legend carries the value at best.pt so no label covers the lines
            legend = f"{run.name}: {values[best]:.3f} (época {int(epochs[best])})"
            ax.plot(epochs, values, color=color, linewidth=2, label=legend)
            ax.plot(
                epochs[best],
                values[best],
                "o",
                color=color,
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                zorder=3,
            )
        ax.set_title(f"{label} en val (● = best.pt)", loc="left", color=TEXT)
        ax.set_xlabel("Época")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.tick_params(colors=MUTED)
    for ax in axes:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    print(args.output)


if __name__ == "__main__":
    main()
