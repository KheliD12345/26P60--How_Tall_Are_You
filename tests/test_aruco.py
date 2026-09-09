from pathlib import Path

import cv2
import numpy as np

import pytest

from height_mvp.aruco import detect_markers, validate_markers
from height_mvp.geometry import (
    calculate_pairwise_geometry,
    estimate_cm_per_pixel,
    estimate_homography,
    transform_point,
)
from height_mvp.models import DetectedMarker, MarkerLayout


def make_layout() -> MarkerLayout:
    return MarkerLayout.from_dict(
        {
            "dictionary": "DICT_4X4_50",
            "marker_size_cm": 18,
            "markers": [
                {"id": marker_id, "name": str(marker_id), "x_cm": marker_id, "y_cm": 0}
                for marker_id in range(4)
            ],
        }
    )


def make_marker_image(path: Path) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((500, 700), 255, dtype=np.uint8)
    for marker_id, (x, y) in enumerate(((50, 50), (500, 50), (50, 300), (500, 300))):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 100)
        image[y : y + 100, x : x + 100] = marker
    assert cv2.imwrite(str(path), image)


def test_detects_marker_corners_and_centres(tmp_path):
    image_path = tmp_path / "markers.png"
    make_marker_image(image_path)

    markers = detect_markers(image_path, make_layout())

    assert [marker.id for marker in markers] == [0, 1, 2, 3]
    assert markers[0].center_x == 99.5
    assert markers[0].center_y == 99.5
    assert markers[0].area_px > 9000


def test_returns_no_markers_for_plain_image(tmp_path):
    image_path = tmp_path / "plain.png"
    assert cv2.imwrite(str(image_path), np.full((200, 200), 255, dtype=np.uint8))

    assert detect_markers(image_path, make_layout()) == ()


def test_rejects_missing_required_markers():
    marker = DetectedMarker(
        id=0,
        corners=((0.0, 0.0),) * 4,
        center_x=0.0,
        center_y=0.0,
        area_px=1.0,
    )

    with pytest.raises(ValueError, match="missing required ArUco markers: 1, 2, 3"):
        validate_markers((marker,), make_layout())


def test_rejects_duplicate_marker_ids():
    marker = DetectedMarker(
        id=0,
        corners=((0.0, 0.0),) * 4,
        center_x=0.0,
        center_y=0.0,
        area_px=1.0,
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_markers((marker, marker), make_layout())


def test_calculates_pairwise_geometry():
    markers = tuple(
        DetectedMarker(
            id=marker_id,
            corners=((0.0, 0.0),) * 4,
            center_x=center[0],
            center_y=center[1],
            area_px=1.0,
        )
        for marker_id, center in enumerate(((0.0, 0.0), (3.0, 4.0), (0.0, 5.0), (3.0, 9.0)))
    )

    geometry = calculate_pairwise_geometry(markers, make_layout())

    assert len(geometry) == 6
    assert geometry[0].first_id == 0
    assert geometry[0].second_id == 1
    assert geometry[0].pixel_delta == (3.0, 4.0)
    assert geometry[0].pixel_distance == 5.0
    assert geometry[0].physical_distance_cm == 1.0


def test_estimates_cm_per_pixel():
    markers = tuple(
        DetectedMarker(
            id=marker_id,
            corners=((0.0, 0.0),) * 4,
            center_x=marker_id * 10.0,
            center_y=0.0,
            area_px=1.0,
        )
        for marker_id in range(4)
    )

    geometry = calculate_pairwise_geometry(markers, make_layout())

    assert estimate_cm_per_pixel(geometry) == 0.1


def test_rejects_geometry_without_a_valid_scale_pair():
    with pytest.raises(ValueError, match="could not estimate scale"):
        estimate_cm_per_pixel(())


def test_estimates_homography_and_transforms_points():
    layout = MarkerLayout.from_dict(
        {
            "dictionary": "DICT_4X4_50",
            "marker_size_cm": 18,
            "markers": [
                {"id": 0, "name": "a", "x_cm": 0, "y_cm": 0},
                {"id": 1, "name": "b", "x_cm": 100, "y_cm": 0},
                {"id": 2, "name": "c", "x_cm": 0, "y_cm": 150},
                {"id": 3, "name": "d", "x_cm": 100, "y_cm": 150},
            ],
        }
    )
    markers = tuple(
        DetectedMarker(
            id=marker_id,
            corners=((0.0, 0.0),) * 4,
            center_x=center[0],
            center_y=center[1],
            area_px=1.0,
        )
        for marker_id, center in enumerate(
            ((50.0, 40.0), (250.0, 40.0), (50.0, 340.0), (250.0, 340.0))
        )
    )

    result = estimate_homography(markers, layout)

    assert result.reprojection_error_cm < 0.0001
    assert transform_point((50.0, 40.0), result.matrix) == pytest.approx((0.0, 0.0))
    assert transform_point((250.0, 340.0), result.matrix) == pytest.approx((100.0, 150.0))


def test_rejects_homography_with_too_few_layout_points():
    layout = MarkerLayout.from_dict(
        {
            "dictionary": "DICT_4X4_50",
            "marker_size_cm": 18,
            "markers": [
                {"id": 0, "name": "a", "x_cm": 0, "y_cm": 0},
                {"id": 1, "name": "b", "x_cm": 1, "y_cm": 0},
                {"id": 2, "name": "c", "x_cm": 0, "y_cm": 1},
            ],
        }
    )

    with pytest.raises(ValueError, match="at least four"):
        estimate_homography((), layout)