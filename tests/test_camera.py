import json

import numpy as np
import pytest

from height_estimation.camera import undistort_image
from height_estimation.models import CameraCalibration


def make_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)),
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def test_camera_calibration_round_trips_json(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(make_calibration().to_dict()), encoding="utf-8")

    assert CameraCalibration.from_json(path) == make_calibration()


def test_undistortion_without_calibration_returns_original_image():
    image = np.full((20, 30, 3), 127, dtype=np.uint8)

    assert undistort_image(image) is image


def test_undistortion_applies_calibration():
    image = np.full((20, 30, 3), 127, dtype=np.uint8)

    result = undistort_image(image, make_calibration())

    assert result.shape == image.shape
    assert result.dtype == image.dtype


@pytest.mark.parametrize(
    "data",
    [
        {"camera_matrix": [[1, 0], [0, 1]], "distortion_coefficients": [0, 0, 0, 0]},
        {"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "distortion_coefficients": [0, 0]},
        {"camera_matrix": [[0, 0, 0], [0, 1, 0], [0, 0, 1]], "distortion_coefficients": [0, 0, 0, 0]},
    ],
)
def test_rejects_invalid_camera_calibration(data):
    with pytest.raises(ValueError):
        CameraCalibration.from_dict(data)