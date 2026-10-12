from yolo_license_plates.training.compare import (
    differing_keys,
    pick_best,
    render_report,
)


def make_run(name: str, imgsz: int, map5095: float) -> dict:
    return {
        "name": name,
        "config": {"name": name, "model": "yolo26n.yaml", "imgsz": imgsz, "batch": 16},
        "epochs_completed": 50,
        "best_epoch": 40,
        "stopped_early": True,
        "train_seconds": 600.0,
        "seconds_per_epoch": 12.0,
        "val_metrics_best": {
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.7,
            "metrics/mAP50(B)": 0.75,
            "metrics/mAP50-95(B)": map5095,
        },
        "hardware": {
            "machine": "aarch64",
            "cpu_count": 4,
            "memory_gb": 6.2,
            "in_docker": True,
        },
        "software": {"python": "3.13", "torch": "2.14.1+cpu", "ultralytics": "8.4.175"},
    }


def test_pick_best_uses_val_map50_95():
    runs = [make_run("a", 416, 0.40), make_run("b", 640, 0.45)]
    assert pick_best(runs)["name"] == "b"


def test_differing_keys_ignores_name():
    runs = [make_run("a", 416, 0.40), make_run("b", 640, 0.45)]
    assert differing_keys(runs) == ["imgsz"]


def test_render_report_names_choice():
    report = render_report([make_run("a", 416, 0.50), make_run("b", 640, 0.45)])
    assert "Configuración elegida: `a`" in report
    assert "Parámetro que cambia: **imgsz**" in report
