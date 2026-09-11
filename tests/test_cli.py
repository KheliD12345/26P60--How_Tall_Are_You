import json

import pytest

from height_estimation.cli import main
from height_estimation.models import (
    CalibrationResult,
    DetectedMarker,
    HomographyResult,
    MarkerPairGeometry,
    PersonEndpoints,
)


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


def make_calibration_result(markers, cm_per_pixel=0.1):
    geometry = tuple(
        MarkerPairGeometry(
            first_id=0,
            second_id=1,
            pixel_delta=(10.0, 0.0),
            physical_delta_cm=(1.0, 0.0),
            pixel_distance=10.0,
            physical_distance_cm=1.0,
        )
        for _ in range(6)
    )
    homography = HomographyResult(
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        reprojection_error_cm=0.0,
    )
    return CalibrationResult(
        markers=markers,
        geometry=geometry,
        cm_per_pixel=cm_per_pixel,
        homography=homography,
    )


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
    calibration = make_calibration_result(markers)
    monkeypatch.setattr(
        "height_estimation.cli.calibrate_image",
        lambda image, layout: calibration,
    )
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
    assert result["person"] is None
    assert "reprojection error" in capsys.readouterr().out


def test_cli_writes_person_endpoints(tmp_path, monkeypatch, capsys):
    markers = tuple(
        DetectedMarker(
            id=marker_id,
            corners=((1.0, 2.0),) * 4,
            center_x=marker_id,
            center_y=marker_id,
            area_px=1.0,
        )
        for marker_id in range(4)
    )
    calibration = CalibrationResult(
        markers=markers,
        geometry=(),
        cm_per_pixel=0.1,
        homography=HomographyResult(
            matrix=((1.0, 0.0, 0.0),) * 3,
            reprojection_error_cm=0.0,
        ),
        person=PersonEndpoints(
            box=(10, 20, 80, 180),
            top_of_head=(50.0, 20.0),
            bottom_of_feet=(50.0, 199.0),
            score=1.23,
        ),
    )
    monkeypatch.setattr(
        "height_estimation.cli.calibrate_image",
        lambda image, layout: calibration,
    )
    image_path = tmp_path / "image.jpg"
    output_path = tmp_path / "detections.json"
    image_path.write_bytes(b"image")

    assert main(
        [str(image_path), "--output", str(output_path)]
    ) == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["person"]["top_of_head"] == [50.0, 20.0]
    assert result["person"]["bottom_of_feet"] == [50.0, 199.0]


def test_cli_writes_requested_overlay(tmp_path, monkeypatch, capsys):
    markers = tuple(
        DetectedMarker(
            id=marker_id,
            corners=((1.0, 2.0),) * 4,
            center_x=marker_id,
            center_y=marker_id,
            area_px=1.0,
        )
        for marker_id in range(4)
    )
    calibration = make_calibration_result(markers)
    calls = []
    monkeypatch.setattr("height_estimation.cli.calibrate_image", lambda image, layout: calibration)
    monkeypatch.setattr(
        "height_estimation.cli.write_calibration_overlay",
        lambda image, result, output: calls.append((image, result, output)),
    )
    image_path = tmp_path / "image.jpg"
    overlay_path = tmp_path / "overlay.png"
    image_path.write_bytes(b"image")

    assert main([str(image_path), "--overlay", str(overlay_path)]) == 0

    assert calls == [(image_path, calibration, overlay_path)]
    assert "Wrote calibration overlay" in capsys.readouterr().out


def test_cli_rejects_missing_markers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "height_estimation.cli.calibrate_image",
        lambda image, layout: (_ for _ in ()).throw(
            ValueError("missing required ArUco markers: 1, 2, 3")
        ),
    )
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")

    with pytest.raises(SystemExit) as error:
        main([str(image_path)])

    assert error.value.code == 2
    assert "missing required ArUco markers" in capsys.readouterr().err
