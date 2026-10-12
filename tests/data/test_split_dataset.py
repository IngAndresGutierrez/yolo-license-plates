import functools
import operator

from yolo_license_plates.data.split_dataset import (
    find_duplicates,
    natural_key,
    split_names,
)


def test_natural_key_orders_numbers_numerically():
    names = ["Cars10.png", "Cars9.png", "Cars100.png"]
    assert sorted(names, key=natural_key) == ["Cars9.png", "Cars10.png", "Cars100.png"]


def test_find_duplicates_keeps_lowest_numbered_copy(tmp_path):
    for name, content in [
        ("Cars27.png", b"same"),
        ("Cars3.png", b"same"),
        ("Cars332.png", b"same"),
        ("Cars5.png", b"unique"),
    ]:
        (tmp_path / name).write_bytes(content)

    duplicates = find_duplicates(sorted(tmp_path.iterdir()))

    assert {k.name: sorted(p.name for p in v) for k, v in duplicates.items()} == {
        "Cars3.png": ["Cars27.png", "Cars332.png"]
    }


def test_split_names_is_deterministic_and_complete():
    names = [f"Cars{i}.png" for i in range(302)]

    first = split_names(names, (0.7, 0.2, 0.1), seed=42)
    second = split_names(list(reversed(names)), (0.7, 0.2, 0.1), seed=42)

    assert first == second
    assert [len(first[s]) for s in ("train", "val", "test")] == [211, 60, 31]
    assert sorted(functools.reduce(operator.iadd, first.values(), [])) == sorted(names)
