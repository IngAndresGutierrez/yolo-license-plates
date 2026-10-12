import pytest

from yolo_license_plates.data.voc_to_yolo import convert_annotation, voc_box_to_yolo

VOC_XML = """<annotation>
    <size><width>500</width><height>268</height><depth>3</depth></size>
    <object>
        <name>licence</name>
        <bndbox><xmin>226</xmin><ymin>125</ymin><xmax>419</xmax><ymax>173</ymax></bndbox>
    </object>
    <object>
        <name>licence</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>500</xmax><ymax>268</ymax></bndbox>
    </object>
</annotation>
"""


def test_voc_box_to_yolo_normalizes_center_and_size():
    assert voc_box_to_yolo((226, 125, 419, 173), 500, 268) == pytest.approx(
        (322.5 / 500, 149 / 268, 193 / 500, 48 / 268)
    )


def test_voc_box_to_yolo_clips_to_image():
    assert voc_box_to_yolo((-10, -5, 510, 300), 500, 268) == pytest.approx(
        (0.5, 0.5, 1.0, 1.0)
    )


def test_voc_box_to_yolo_rejects_empty_box():
    with pytest.raises(ValueError):
        voc_box_to_yolo((100, 50, 100, 80), 500, 268)


def test_convert_annotation_writes_one_line_per_object(tmp_path):
    xml_path = tmp_path / "Cars0.xml"
    xml_path.write_text(VOC_XML)

    assert convert_annotation(xml_path) == [
        "0 0.645000 0.555970 0.386000 0.179104",
        "0 0.500000 0.500000 1.000000 1.000000",
    ]


def test_convert_annotation_rejects_unknown_class(tmp_path):
    xml_path = tmp_path / "Cars0.xml"
    xml_path.write_text(VOC_XML.replace("licence", "car", 1))

    with pytest.raises(ValueError, match="unknown class"):
        convert_annotation(xml_path)
