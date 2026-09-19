from pathlib import Path

import cv2
import numpy as np
import pytest

from height_estimation.calibration import (
    calibrate_image,
    load_calibration,
    save_calibration,
)
from height_estimation.models import (
    CalibrationResult,
    CameraCalibration,
    HomographyResult,
    MarkerLayout,
    PersonEndpoints,
)


def make_layout() -> MarkerLayout:
    return MarkerLayout.from_dict(
        {
            "dictionary": "DICT_4X4_50",
            "marker_size_cm": 18,
            "markers": [
                {"id": 0, "name": "bottom_left", "x_cm": 0, "y_cm": 0},
                {"id": 1, "name": "bottom_right", "x_cm": 100, "y_cm": 0},
                {"id": 2, "name": "top_left", "x_cm": 0, "y_cm": 60},
                {"id": 3, "name": "top_right", "x_cm": 100, "y_cm": 60},
            ],
        }
    )


def make_image(path: Path) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((500, 700), 255, dtype=np.uint8)
    for marker_id, (x, y) in enumerate(((50, 350), (550, 350), (50, 50), (550, 50))):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 100)
        image[y : y + 100, x : x + 100] = marker
    assert cv2.imwrite(str(path), image)


def test_calibrates_image_in_one_result(tmp_path):
    image_path = tmp_path / "markers.png"
    make_image(image_path)

    result = calibrate_image(image_path, make_layout())

    assert [marker.id for marker in result.markers] == [0, 1, 2, 3]
    assert len(result.geometry) == 6
    assert result.cm_per_pixel == 0.2
    assert result.homography.reprojection_error_cm < 0.0001
    assert result.to_dict()["scale"]["pixels_per_cm"] == 5.0
    assert result.person is None


def test_calibrates_person_height(tmp_path, monkeypatch):
    image_path = tmp_path / "markers.png"
    make_image(image_path)
    person = PersonEndpoints(
        box=(200, 100, 100, 300),
        top_of_head=(250.0, 100.0),
        bottom_of_feet=(250.0, 300.0),
        score=1.5,
    )
    monkeypatch.setattr(
        "height_estimation.calibration.detect_person_endpoints",
        lambda image: person,
    )

    result = calibrate_image(image_path, make_layout())

    assert result.person is person
    assert result.to_dict()["person"]["height_px"] == 200.0
    assert result.to_dict()["person"]["height_cm"] == 40.0
    assert result.to_dict()["person"]["perspective_height_cm"] == 40.0


def test_skips_perspective_height_when_validation_fails(tmp_path, monkeypatch):
    image_path = tmp_path / "markers.png"
    make_image(image_path)
    person = PersonEndpoints(
        box=(200, 100, 100, 300),
        top_of_head=(250.0, 100.0),
        bottom_of_feet=(250.0, 300.0),
        score=1.5,
    )
    monkeypatch.setattr(
        "height_estimation.calibration.detect_person_endpoints",
        lambda image: person,
    )
    monkeypatch.setattr(
        "height_estimation.calibration.validate_perspective_result",
        lambda *args: (_ for _ in ()).throw(ValueError("not reliable")),
    )

    result = calibrate_image(image_path, make_layout())

    assert result.person is person
    assert result.perspective_height_cm is None
    assert "perspective_height_cm" not in result.to_dict()["person"]


def test_calibration_result_includes_scene_recovery_metadata(tmp_path):
    image_path = tmp_path / "markers.png"
    make_image(image_path)
    camera_calibration = CameraCalibration(
        camera_matrix=((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)),
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    result = calibrate_image(
        image_path,
        make_layout(),
        camera_calibration=camera_calibration,
    )

    payload = result.to_dict()
    assert result.camera_calibration is camera_calibration
    assert payload["camera_calibration"] == camera_calibration.to_dict()
    assert payload["diagnostics"] == [
        "Camera undistortion applied before marker detection."
    ]


def test_calibration_result_rejects_invalid_scale():
    with pytest.raises(ValueError, match="cm_per_pixel"):
        CalibrationResult(
            markers=(),
            geometry=(),
            cm_per_pixel=0.0,
            homography=HomographyResult(
                matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                reprojection_error_cm=0.0,
            ),
        )


def test_calibration_result_persists_and_loads(tmp_path):
    image_path = tmp_path / "markers.png"
    calibration_path = tmp_path / "nested" / "calibration.json"
    make_image(image_path)

    result = calibrate_image(image_path, make_layout())
    save_calibration(result, calibration_path)
    loaded = load_calibration(calibration_path)

    assert loaded.to_dict() == result.to_dict()


def test_calibration_result_persists_person_data(tmp_path, monkeypatch):
    image_path = tmp_path / "markers.png"
    calibration_path = tmp_path / "calibration.json"
    make_image(image_path)
    person = PersonEndpoints(
        box=(200, 100, 100, 300),
        top_of_head=(250.0, 100.0),
        bottom_of_feet=(250.0, 300.0),
        score=1.5,
    )
    monkeypatch.setattr(
        "height_estimation.calibration.detect_person_endpoints",
        lambda image: person,
    )

    result = calibrate_image(image_path, make_layout())
    save_calibration(result, calibration_path)
    loaded = load_calibration(calibration_path)

    assert loaded.person == person
    assert loaded.perspective_height_cm == result.perspective_height_cm
    assert loaded.to_dict() == result.to_dict()