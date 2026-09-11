from pathlib import Path

import pytest

from height_estimation.models import DetectedMarker, MarkerLayout, PersonEndpoints


LAYOUT_PATH = Path(__file__).parents[1] / "configs" / "marker_layout.json"


def test_loads_backdrop_layout():
    layout = MarkerLayout.from_json(LAYOUT_PATH)

    assert layout.dictionary == "DICT_4X4_50"
    assert layout.marker_size_cm == 18.0
    assert layout.marker_ids == (0, 1, 2, 3)


def test_rejects_duplicate_marker_ids():
    data = {
        "dictionary": "DICT_4X4_50",
        "marker_size_cm": 18,
        "markers": [
            {"id": 0, "name": "left", "x_cm": 0, "y_cm": 0},
            {"id": 0, "name": "right", "x_cm": 90, "y_cm": 0},
        ],
    }

    with pytest.raises(ValueError, match="unique"):
        MarkerLayout.from_dict(data)


def test_serializes_detected_marker():
    marker = DetectedMarker(
        id=4,
        corners=((1.123, 2.456), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)),
        center_x=4.0,
        center_y=5.0,
        area_px=36.0,
    )

    assert marker.to_dict()["center"] == [4.0, 5.0]
    assert marker.to_dict()["corners"][0] == [1.12, 2.46]


def test_serializes_person_endpoints():
    person = PersonEndpoints(
        box=(10, 20, 80, 180),
        top_of_head=(50.125, 22.456),
        bottom_of_feet=(50.875, 199.987),
        score=1.2345,
    )

    assert person.to_dict() == {
        "box": [10, 20, 80, 180],
        "top_of_head": [50.12, 22.46],
        "bottom_of_feet": [50.88, 199.99],
        "score": 1.23,
    }
