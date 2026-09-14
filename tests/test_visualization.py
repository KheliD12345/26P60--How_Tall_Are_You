from pathlib import Path

import cv2
import numpy as np

from height_estimation.models import (
    CalibrationResult,
    DetectedMarker,
    HomographyResult,
    MarkerPairGeometry,
    PersonEndpoints,
)
from height_estimation.visualization import write_calibration_overlay


def make_calibration() -> CalibrationResult:
    markers = (
        DetectedMarker(
            id=0,
            corners=((20.0, 20.0), (40.0, 20.0), (40.0, 40.0), (20.0, 40.0)),
            center_x=30.0,
            center_y=30.0,
            area_px=400.0,
        ),
        DetectedMarker(
            id=1,
            corners=((60.0, 20.0), (80.0, 20.0), (80.0, 40.0), (60.0, 40.0)),
            center_x=70.0,
            center_y=30.0,
            area_px=400.0,
        ),
    )
    pair = MarkerPairGeometry(
        first_id=0,
        second_id=1,
        pixel_delta=(40.0, 0.0),
        physical_delta_cm=(10.0, 0.0),
        pixel_distance=40.0,
        physical_distance_cm=10.0,
    )
    homography = HomographyResult(
        matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        reprojection_error_cm=0.0,
    )
    return CalibrationResult(
        markers=markers,
        geometry=(pair,),
        cm_per_pixel=0.25,
        homography=homography,
    )


def test_writes_calibration_overlay(tmp_path):
    image_path = tmp_path / "image.png"
    output_path = tmp_path / "nested" / "overlay.png"
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    write_calibration_overlay(image_path, make_calibration(), output_path)

    overlay = cv2.imread(str(output_path))
    assert overlay is not None
    assert overlay.shape == image.shape
    assert np.any(overlay != 255)


def test_adds_person_endpoint_annotations(tmp_path):
    image_path = tmp_path / "image.png"
    without_person_path = tmp_path / "without-person.png"
    with_person_path = tmp_path / "with-person.png"
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    write_calibration_overlay(
        image_path,
        make_calibration(),
        without_person_path,
    )
    calibration = make_calibration()
    calibration = CalibrationResult(
        markers=calibration.markers,
        geometry=calibration.geometry,
        cm_per_pixel=calibration.cm_per_pixel,
        homography=calibration.homography,
        person=PersonEndpoints(
            box=(10, 10, 50, 80),
            top_of_head=(35.0, 10.0),
            bottom_of_feet=(35.0, 89.0),
            score=1.2,
        ),
    )
    write_calibration_overlay(image_path, calibration, with_person_path)

    without_person = cv2.imread(str(without_person_path))
    with_person = cv2.imread(str(with_person_path))
    assert without_person is not None
    assert with_person is not None
    assert np.any(without_person != with_person)


def test_rejects_missing_overlay_input(tmp_path):
    with np.testing.assert_raises(ValueError):
        write_calibration_overlay(
            tmp_path / "missing.png",
            make_calibration(),
            tmp_path / "overlay.png",
        )