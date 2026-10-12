"""Compare training runs on the validation split and pick the best one.

Reads <run>/run_info.json from each run directory and writes a Markdown
report. The choice uses val mAP50-95 of best.pt; the test split is not used.
"""

import argparse
import json
from pathlib import Path

SELECTION_METRIC = "metrics/mAP50-95(B)"
METRICS = {
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}
CONFIG_ROWS = ("model", "imgsz", "batch", "epochs", "patience", "seed", "device")


def load_runs(run_dirs: list[Path]) -> list[dict]:
    runs = []
    for run_dir in run_dirs:
        info_path = run_dir / "run_info.json"
        if not info_path.is_file():
            raise SystemExit(f"Missing {info_path}; did the run finish?")
        runs.append(json.loads(info_path.read_text()))
    return runs


def differing_keys(runs: list[dict]) -> list[str]:
    """Config keys whose values differ between runs (ignoring the name)."""
    keys = set().union(*(r["config"] for r in runs)) - {"name", "description"}
    return sorted(k for k in keys if len({repr(r["config"].get(k)) for r in runs}) > 1)


def pick_best(runs: list[dict]) -> dict:
    return max(runs, key=lambda r: r["val_metrics_best"][SELECTION_METRIC])


def render_report(runs: list[dict]) -> str:
    best = pick_best(runs)
    changed = differing_keys(runs)
    header = "| | " + " | ".join(r["name"] for r in runs) + " |"
    divider = "|---|" + "---|" * len(runs)

    def row(label: str, values: list) -> str:
        return f"| {label} | " + " | ".join(str(v) for v in values) + " |"

    lines = [
        "# Comparación de configuraciones (validación)",
        "",
        (
            f"Parámetro que cambia: **{', '.join(changed) or 'ninguno'}**. "
            f"Criterio: {METRICS[SELECTION_METRIC]} de `best.pt` en el split **val**."
        ),
        "",
        header,
        divider,
        *(row(f"`{k}`", [r["config"].get(k) for r in runs]) for k in CONFIG_ROWS),
        row("Épocas completadas", [r["epochs_completed"] for r in runs]),
        row("Mejor época", [r["best_epoch"] for r in runs]),
        row("Parada temprana", ["sí" if r["stopped_early"] else "no" for r in runs]),
        row("Tiempo total (min)", [round(r["train_seconds"] / 60, 1) for r in runs]),
        row("Segundos por época", [r["seconds_per_epoch"] for r in runs]),
        *(
            row(f"**{label}**", [r["val_metrics_best"][key] for r in runs])
            for key, label in METRICS.items()
        ),
        "",
        (
            f"**Configuración elegida: `{best['name']}`** "
            f"({METRICS[SELECTION_METRIC]} val = {best['val_metrics_best'][SELECTION_METRIC]})."
        ),
        "",
        (
            "El tiempo se mide con `time.perf_counter()`, que en macOS no cuenta el "
            "tiempo en reposo del equipo. La columna `time` de `results.csv` es tiempo "
            "de reloj y sí lo incluye."
        ),
        "",
        "## Hardware y software",
        "",
    ]
    for r in runs:
        hw, sw = r["hardware"], r["software"]
        cores = hw.get("core_types")
        core_text = (
            f" ({cores['performance']} de rendimiento + {cores['efficiency']} de eficiencia)"
            if cores
            else ""
        )
        lines.append(
            f"- `{r['name']}`: {hw.get('processor') or hw['machine']}, "
            f"{hw['cpu_count']} CPUs{core_text}, {hw.get('torch_threads')} hilos de torch, "
            f"{hw['memory_gb']} GB RAM, Docker={hw['in_docker']}; "
            f"Python {sw['python']}, torch {sw['torch']}, Ultralytics {sw['ultralytics']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Run directories to compare")
    parser.add_argument("--output", type=Path, default=Path("reports/comparacion.md"))
    args = parser.parse_args(argv)

    runs = load_runs(args.runs)
    report = render_report(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
