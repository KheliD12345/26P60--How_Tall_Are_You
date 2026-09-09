import json

import pytest

from height_mvp.cli import main
from height_mvp.models import DetectedMarker


def make_layout_data():
    return {
        "dictionary": "DICT_4X4_50",
        "marker_size_cm": 18,
        "markers": [
            {"id": 0, "name": "0", "x_cm": 0, "y_cm": 0},
            {"id": 1, "name": "1", "x_cm": 1, "y_cm": 0},
            {"id": 2, "name": "2", "x_cm": 0, "y_cm": 1},
            {"id": 3, "name": "3", "x_cm": 1, "y_cm": 1},
        ],
    }


def test_cli_loads_default_layout(capsys):
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "Loaded 4 markers" in output
    assert "DICT_4X4_50" in output


def test_cli_writes_detection_json(tmp_path, monkeypatch, capsys):
    markers = tuple(
        DetectedMarker(
            id=marker_id,
            corners=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)),
            center_x=(marker_id % 2) * 10.0,
            center_y=(marker_id // 2) * 10.0,
            area_px=4.0,
        )
        for marker_id in range(4)
    )
    monkeypatch.setattr("height_mvp.cli.detect_markers", lambda image, layout: markers)
    image_path = tmp_path / "image.jpg"
    layout_path = tmp_path / "layout.json"
    output_path = tmp_path / "detections.json"
    image_path.write_bytes(b"image")
    layout_path.write_text(json.dumps(make_layout_data()), encoding="utf-8")

    assert main(
        [
            str(image_path),
            "--layout",
            str(layout_path),
            "--output",
            str(output_path),
        ]
    ) == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["markers"][0]["id"] == 0
    assert len(result["geometry"]) == 6
    assert result["scale"]["cm_per_pixel"] == 0.1
    assert result["homography"]["reprojection_error_cm"] < 0.0001
    assert "reprojection error" in capsys.readouterr().out


def test_cli_rejects_missing_markers(tmp_path, monkeypatch, capsys):
    marker = DetectedMarker(
        id=0,
        corners=((1.0, 2.0),) * 4,
        center_x=1.0,
        center_y=2.0,
        area_px=1.0,
    )
    monkeypatch.setattr("height_mvp.cli.detect_markers", lambda image, layout: (marker,))
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")

    with pytest.raises(SystemExit) as error:
        main([str(image_path)])

    assert error.value.code == 2
    assert "missing required ArUco markers" in capsys.readouterr().err
