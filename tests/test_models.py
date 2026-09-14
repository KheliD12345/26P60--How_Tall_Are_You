from pathlib import Path

import pytest

from height_estimation.height import calculate_height_cm
from height_estimation.models import (
    CalibrationResult,
    DetectedMarker,
    HomographyResult,
    MarkerLayout,
    PersonEndpoints,
)


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
        "height_px": 177.53,
        "score": 1.23,
    }


def test_calculates_height_from_pixel_height():
    assert calculate_height_cm(179.0, 0.1) == pytest.approx(17.9)


def test_baseline_height_stays_separate_from_homography():
    person = PersonEndpoints(
        box=(10, 20, 80, 180),
        top_of_head=(50.0, 20.0),
        bottom_of_feet=(50.0, 199.0),
        score=1.0,
    )
    calibration = CalibrationResult(
        markers=(),
        geometry=(),
        cm_per_pixel=0.1,
        homography=HomographyResult(
            matrix=((1.0, 0.0, 25.0), (0.0, 1.5, -10.0), (0.001, 0.0, 1.0)),
            reprojection_error_cm=0.0,
        ),
        person=person,
    )

    result = calibration.to_dict()

    assert result["person"]["height_px"] == 179.0
    assert result["person"]["height_cm"] == 17.9


@pytest.mark.parametrize(
    "box",
    [(-1, 20, 80, 180), (10, 20, 0, 180), (10, 20, 80, 0)],
)
def test_rejects_invalid_person_boxes(box):
    with pytest.raises(ValueError, match="person box"):
        PersonEndpoints(
            box=box,
            top_of_head=(50.0, 20.0),
            bottom_of_feet=(50.0, 199.0),
            score=1.0,
        )


def test_rejects_non_finite_person_values():
    with pytest.raises(ValueError, match="finite coordinates"):
        PersonEndpoints(
            box=(10, 20, 80, 180),
            top_of_head=(float("nan"), 20.0),
            bottom_of_feet=(50.0, 199.0),
            score=1.0,
        )

    with pytest.raises(ValueError, match="finite"):
        PersonEndpoints(
            box=(10, 20, 80, 180),
            top_of_head=(50.0, 20.0),
            bottom_of_feet=(50.0, 199.0),
            score=float("inf"),
        )


def test_rejects_reversed_person_endpoints():
    with pytest.raises(ValueError, match="above"):
        PersonEndpoints(
            box=(10, 20, 80, 180),
            top_of_head=(50.0, 200.0),
            bottom_of_feet=(50.0, 199.0),
            score=1.0,
        )
