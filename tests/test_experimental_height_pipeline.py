from height_estimation.frame_stick import (
    MarkerDetection,
    SubjectEndpoints,
    estimate_height,
)
from height_estimation.models import MarkerLayout


def _marker(marker_id: int, center: tuple[float, float]) -> MarkerDetection:
    x, y = center
    corners = (
        (x - 4.0, y - 4.0),
        (x + 4.0, y - 4.0),
        (x + 4.0, y + 4.0),
        (x - 4.0, y + 4.0),
    )
    return MarkerDetection(marker_id, corners, center, 64.0)


def _layout() -> MarkerLayout:
    return MarkerLayout.from_dict(
        {
            "dictionary": "DICT_4X4_50",
            "marker_size_cm": 18.0,
            "markers": [
                {"id": 0, "name": "top_left", "x_cm": 0.0, "y_cm": 194.0},
                {"id": 1, "name": "top_right", "x_cm": 103.5, "y_cm": 194.0},
                {"id": 2, "name": "bottom_left", "x_cm": 0.0, "y_cm": 0.0},
                {"id": 3, "name": "bottom_right", "x_cm": 103.5, "y_cm": 0.0},
            ],
        }
    )


def _frame_markers(layout: MarkerLayout) -> dict[int, MarkerDetection]:
    return {
        marker.id: _marker(
            marker.id,
            (100.0 + 2.0 * marker.x_cm, 500.0 - 2.0 * marker.y_cm),
        )
        for marker in layout.markers
    }


def _stick_markers() -> tuple[MarkerDetection, MarkerDetection]:
    return _marker(4, (420.0, 100.0)), _marker(5, (420.0, 181.0))


def test_frame_homography_is_primary_and_stick_scale_is_independent():
    layout = _layout()
    subject = SubjectEndpoints(
        head_top=(200.0, 300.0),
        heel_points=((201.0, 100.0),),
        confidence=0.9,
    )

    result = estimate_height(_frame_markers(layout), _stick_markers(), subject, layout)

    assert result["method"] == "four_frame_marker_homography"
    assert result["height_cm"] == 100.0
    assert result["stick_scale_check"]["height_cm"] == 200.0
    assert result["stick_scale_check"]["difference_from_frame_homography_cm"] == 100.0


def test_frame_homography_succeeds_without_stick_markers():
    layout = _layout()
    subject = SubjectEndpoints(
        head_top=(200.0, 300.0),
        heel_points=((201.0, 100.0),),
        confidence=0.9,
    )

    result = estimate_height(_frame_markers(layout), None, subject, layout)

    assert result["method"] == "four_frame_marker_homography"
    assert result["height_cm"] == 100.0
    assert result["stick_scale_check"] is None


def test_stick_scale_is_fallback_when_frame_markers_are_incomplete():
    subject = SubjectEndpoints(
        head_top=(200.0, 300.0),
        heel_points=((201.0, 100.0),),
        confidence=0.9,
    )

    result = estimate_height({}, _stick_markers(), subject, _layout())

    assert result["method"] == "stick_marker_scalar_fallback"
    assert result["height_cm"] == 200.0
    assert result["frame_homography"] is None
