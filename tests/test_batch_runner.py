import json

import numpy as np
import pytest
from openpyxl.cell.cell import MergedCell
from openpyxl import load_workbook

from height_estimation import excel_runner, json_runner
from height_estimation.frame_stick import MarkerDetection, SubjectEndpoints


def _layout_data():
    return {
        "dictionary": "DICT_4X4_50",
        "marker_size_cm": 18.0,
        "stick_reference_distance_cm": 81.0,
        "markers": [
            {"id": 0, "name": "top_left", "x_cm": 0.0, "y_cm": 191.5},
            {"id": 1, "name": "top_right", "x_cm": 105.5, "y_cm": 191.5},
            {"id": 2, "name": "bottom_left", "x_cm": 0.0, "y_cm": 0.0},
            {"id": 3, "name": "bottom_right", "x_cm": 105.5, "y_cm": 0.0},
        ],
    }


def test_discover_images_sorts_fallback_ids_and_preserves_participant_labels(
    tmp_path,
):
    random_dir = tmp_path / "random"
    random_dir.mkdir()
    (random_dir / "b-photo.jpg").touch()
    (random_dir / "A-photo.png").touch()

    random_selections = excel_runner.discover_images(random_dir)

    assert [(item.participant_id, item.path.name) for item in random_selections] == [
        ("P001", "A-photo.png"),
        ("P002", "b-photo.jpg"),
    ]

    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    (mixed_dir / "a-photo.jpg").touch()
    (mixed_dir / "P001_2.jpg").touch()
    (mixed_dir / "z-photo.jpg").touch()

    mixed_selections = excel_runner.discover_images(mixed_dir)

    assert [(item.participant_id, item.path.name) for item in mixed_selections] == [
        ("P002", "a-photo.jpg"),
        ("P001", "P001_2.jpg"),
        ("P003", "z-photo.jpg"),
    ]
    (mixed_dir / "P0001-copy.jpg").touch()
    with pytest.raises(ValueError, match="duplicate participant ID P001"):
        excel_runner.discover_images(mixed_dir)


def test_excel_and_json_commands_write_both_estimates_from_shared_pipeline(
    tmp_path,
    monkeypatch,
    capsys,
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "b-photo.jpg").touch()
    (input_dir / "A-photo.jpg").touch()
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(_layout_data()), encoding="utf-8")
    monkeypatch.setattr(excel_runner, "DEFAULT_LAYOUT", layout_path)
    monkeypatch.setattr(
        excel_runner,
        "build_body_detector",
        lambda: object(),
    )

    def read_image(path):
        assert path
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(excel_runner.cv2, "imread", read_image)
    subject = SubjectEndpoints((1.0, 1.0), ((1.0, 2.0),), 0.9)

    def detect_subject(detector, image):
        assert detector is not None
        assert image.size
        return subject, None

    def detect_markers(image, dictionary_name, marker_ids):
        assert image.size
        assert dictionary_name == "DICT_4X4_50"
        return {
            marker_id: MarkerDetection(
                marker_id,
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                (0.5, 0.5),
                1.0,
            )
            for marker_id in marker_ids
        }

    def estimate_height(frame, stick, endpoints, layout):
        assert endpoints is subject
        assert layout.dictionary == "DICT_4X4_50"
        assert stick is None or len(stick) == 2
        return {"height_cm": 172.25 if frame else 169.5}

    monkeypatch.setattr(excel_runner, "detect_subject", detect_subject)
    monkeypatch.setattr(excel_runner, "detect_markers", detect_markers)
    monkeypatch.setattr(excel_runner, "estimate_height", estimate_height)

    workbook_path = tmp_path / "outputs" / "Participant Heights.xlsx"
    assert excel_runner.run_batch(workbook_path, input_dir) == 0

    workbook = load_workbook(workbook_path)
    sheet = workbook.active
    assert sheet is not None
    assert [sheet.cell(row, 1).value for row in (2, 3)] == ["P001", "P002"]
    assert [sheet.cell(row, 3).value for row in (2, 3)] == [172.25, 172.25]
    assert [sheet.cell(row, 4).value for row in (2, 3)] == [169.5, 169.5]
    measured_cell = sheet.cell(2, 2)
    assert not isinstance(measured_cell, MergedCell)
    measured_cell.value = 165.0
    workbook.save(workbook_path)
    workbook.close()
    assert excel_runner.run_batch(workbook_path, input_dir) == 0
    workbook = load_workbook(workbook_path)
    sheet = workbook.active
    assert sheet is not None
    assert sheet.cell(2, 2).value == 165.0
    workbook.close()

    output_dir = tmp_path / "json"
    assert json_runner.run_batch(input_dir, output_dir) == 0
    results = [
        json.loads((output_dir / f"P{number:03d}.json").read_text(encoding="utf-8"))
        for number in (1, 2)
    ]
    assert [result["frame_estimated_height_cm"] for result in results] == [
        172.25,
        172.25,
    ]
    assert [result["stick_estimated_height_cm"] for result in results] == [
        169.5,
        169.5,
    ]
    assert [result["geometric_height_cm"] for result in results] == [172.25, 172.25]
    assert "HOG fallback disabled" in capsys.readouterr().out
