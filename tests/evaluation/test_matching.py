import pytest

from yolo_license_plates.evaluation.predict_gallery import (
    Box,
    ImageResult,
    classify,
    iou,
    match,
    read_yolo_labels,
)

PLATE = Box(100, 100, 200, 150)


def test_iou_identical_and_disjoint():
    assert iou(PLATE, PLATE) == pytest.approx(1.0)
    assert iou(PLATE, Box(300, 300, 350, 320)) == 0.0


def test_iou_half_overlap():
    # Same height, shifted by half the width: inter = 50*50, union = 150*50
    assert iou(PLATE, Box(150, 100, 250, 150)) == pytest.approx(1 / 3)


def test_match_hit_false_positive_and_miss():
    predictions = [Box(102, 101, 198, 149, conf=0.9), Box(400, 10, 450, 30, conf=0.6)]
    ground_truth = [PLATE, Box(300, 200, 380, 230)]

    assert match(predictions, ground_truth, 0.5) == ([0], [1], [1])


def test_match_duplicate_detection_counts_as_false_positive():
    predictions = [Box(100, 100, 200, 150, conf=0.5), Box(101, 100, 201, 150, conf=0.9)]

    true_positives, false_positives, missed = match(predictions, [PLATE], 0.5)

    # The higher-confidence box takes the plate; the other is a duplicate
    assert (true_positives, false_positives, missed) == ([1], [0], [])


@pytest.mark.parametrize(
    ("predictions", "fp", "missed", "status"),
    [
        ([], [], [0], "no_detections"),
        ([PLATE], [], [], "correct"),
        ([PLATE], [0], [], "false_positive"),
        ([PLATE], [], [0], "missed"),
        ([PLATE], [0], [0], "mixed"),
    ],
)
def test_classify(predictions, fp, missed, status):
    result = ImageResult(
        "x.png", predictions, [PLATE], false_positives=fp, missed=missed
    )
    assert classify(result) == status


def test_read_yolo_labels_converts_to_pixels(tmp_path):
    label = tmp_path / "x.txt"
    label.write_text("0 0.5 0.5 0.2 0.1\n")

    (box,) = read_yolo_labels(label, 500, 200)

    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((200, 90, 300, 110))
