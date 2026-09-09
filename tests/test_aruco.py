from pathlib import Path

import cv2
import numpy as np

import pytest

from height_mvp.aruco import detect_markers, validate_markers
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