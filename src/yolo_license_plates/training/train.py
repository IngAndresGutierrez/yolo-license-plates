"""Train a YOLO license-plate detector from random weights.

Each experiment is a YAML file in configs/experiments/ so it can be repeated
with a single command:

    python -m yolo_license_plates.training.train --config configs/experiments/a_imgsz416.yaml

Command-line flags override the values in the config. Weights, curves and a
run_info.json (hyperparameters, time, hardware, versions) are written to
<project>/<name>/.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "train",
    "model": "yolo26n.yaml",
    "data": "dataset/data.yaml",
    "project": "runs",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 30,
    "device": "cpu",
    "workers": 2,
    "seed": 42,
    "deterministic": True,
    "cache": False,
}

# Keys that are ours, not Ultralytics train() arguments
RUN_KEYS = {"name", "model", "data", "project", "description"}


def load_config(path: Path | None, overrides: dict) -> dict:
    """Merge DEFAULTS, the experiment YAML and non-None CLI overrides."""
    config = dict(DEFAULTS)
    if path is not None:
        config.update(yaml.safe_load(path.read_text()) or {})
    config.update({k: v for k, v in overrides.items() if v is not None})

    if not str(config["model"]).endswith((".yaml", ".yml")):
        raise ValueError(
            f"model must be an architecture YAML to start from random weights, "
            f"got {config['model']!r} (a .pt file carries trained weights)"
        )
    return config


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def describe_hardware() -> dict:
    """Hardware visible to this process (inside Docker: the container's view)."""
    import torch

    memory_gb = None
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        memory_gb = round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9, 1
        )

    processor = platform.processor() or None
    core_types = None
    if platform.system() == "Darwin":
        # platform.processor() only says "arm" on Apple silicon
        processor = _sysctl("machdep.cpu.brand_string") or processor
        performance = _sysctl("hw.perflevel0.physicalcpu")
        efficiency = _sysctl("hw.perflevel1.physicalcpu")
        if performance and efficiency:
            core_types = {
                "performance": int(performance),
                "efficiency": int(efficiency),
            }

    return {
        "machine": platform.machine(),
        "processor": processor,
        "cpu_count": os.cpu_count(),
        "core_types": core_types,
        "torch_threads": torch.get_num_threads(),
        "memory_gb": memory_gb,
        "os": platform.platform(),
        "in_docker": Path("/.dockerenv").exists(),
    }


def describe_software() -> dict:
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }


def read_epoch_log(results_csv: Path) -> list[dict]:
    """Per-epoch rows written by Ultralytics (column names are space-padded)."""
    if not results_csv.is_file():
        return []
    import csv

    with results_csv.open() as f:
        return [
            {k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)
        ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO license-plate detector from random weights.",
    )
    parser.add_argument(
        "--config", type=Path, help="Experiment YAML (configs/experiments/)"
    )
    parser.add_argument("--name", help="Run name inside --project")
    parser.add_argument("--model", help="Architecture YAML, e.g. yolo26n.yaml")
    parser.add_argument("--data", help="Ultralytics dataset YAML")
    parser.add_argument("--project", help="Output directory for runs")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", help="cpu, 0, 0,1...")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.set_defaults(**{k: None for k in DEFAULTS})
    parser.epilog = f"Defaults when neither config nor flag sets a value: {DEFAULTS}"
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    try:
        config = load_config(args.config, overrides)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if not Path(config["data"]).is_file():
        raise SystemExit(f"Dataset config not found: {config['data']}")

    # Imported lazily so --help doesn't pay torch's import cost
    from ultralytics import YOLO

    model = YOLO(config["model"])
    # A model built from an architecture YAML has no checkpoint behind it
    weights_check = {
        "model_source": config["model"],
        "ckpt_is_empty": not model.ckpt,
        "ckpt_path": model.ckpt_path,
        "pretrained_arg": False,
        "parameters": sum(p.numel() for p in model.model.parameters()),
    }
    if not weights_check["ckpt_is_empty"] or model.ckpt_path is not None:
        raise SystemExit(
            f"Refusing to train: model has pretrained weights {weights_check}"
        )

    train_args = {k: v for k, v in config.items() if k not in RUN_KEYS}
    project = Path(config["project"]).resolve()
    started_at = datetime.now(UTC)
    t0 = time.perf_counter()
    metrics = model.train(
        data=config["data"],
        project=str(project),
        name=config["name"],
        exist_ok=False,
        pretrained=False,
        **train_args,
    )
    train_seconds = time.perf_counter() - t0

    # Relative to the working dir so paths are valid both in Docker and on the host
    save_dir = Path(model.trainer.save_dir).relative_to(Path.cwd())
    epochs_log = read_epoch_log(save_dir / "results.csv")
    best_row = max(
        epochs_log, key=lambda r: r.get("metrics/mAP50-95(B)", 0), default={}
    )

    run_info = {
        "name": config["name"],
        "description": config.get("description"),
        "config_file": str(args.config) if args.config else None,
        "config": config,
        "started_at": started_at.isoformat(timespec="seconds"),
        "train_seconds": round(train_seconds, 1),
        "epochs_requested": config["epochs"],
        "epochs_completed": len(epochs_log),
        "seconds_per_epoch": round(train_seconds / len(epochs_log), 1)
        if epochs_log
        else None,
        "best_epoch": int(best_row["epoch"]) if best_row else None,
        "stopped_early": len(epochs_log) < config["epochs"],
        # Final validation of best.pt on the val split (Ultralytics runs it after training)
        "val_metrics_best": {
            k: round(float(v), 4) for k, v in metrics.results_dict.items()
        },
        "weights_check": weights_check,
        "hardware": describe_hardware(),
        "software": describe_software(),
        "outputs": {
            "best_weights": str(save_dir / "weights" / "best.pt"),
            "last_weights": str(save_dir / "weights" / "last.pt"),
            "results_csv": str(save_dir / "results.csv"),
            "curves": sorted(p.name for p in save_dir.glob("*.png")),
        },
    }
    (save_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: run_info[k]
                for k in (
                    "name",
                    "train_seconds",
                    "epochs_completed",
                    "best_epoch",
                    "val_metrics_best",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
