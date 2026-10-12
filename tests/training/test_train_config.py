import pytest

from yolo_license_plates.training.train import DEFAULTS, load_config


def test_load_config_merges_defaults_file_and_overrides(tmp_path):
    config_path = tmp_path / "exp.yaml"
    config_path.write_text("name: exp\nimgsz: 416\nbatch: 8\n")

    config = load_config(config_path, {"batch": 4, "epochs": None})

    assert config["name"] == "exp"
    assert config["imgsz"] == 416
    assert config["batch"] == 4
    assert config["epochs"] == DEFAULTS["epochs"]


@pytest.mark.parametrize("model", ["yolo26n.pt", "runs/a/weights/best.pt"])
def test_load_config_rejects_weight_files(model):
    with pytest.raises(ValueError, match="random weights"):
        load_config(None, {"model": model})


def test_experiment_configs_differ_only_in_imgsz():
    from pathlib import Path

    import yaml

    root = Path(__file__).parents[2] / "configs" / "experiments"
    a = yaml.safe_load((root / "a_imgsz416.yaml").read_text())
    b = yaml.safe_load((root / "b_imgsz640.yaml").read_text())
    ignored = {"name", "description"}
    diff = {k for k in a.keys() | b.keys() if k not in ignored and a.get(k) != b.get(k)}

    assert diff == {"imgsz"}
