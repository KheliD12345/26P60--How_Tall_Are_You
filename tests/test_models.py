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
from height_estimation.advanced_models import (
    BodyDetections,
    HeightEstimate,
    Landmark,
    MeasurementMethod,
    MeasurementResult,
    QualityAssessment,
    QualityLevel,
    QualityMetrics,
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


def test_landmark_converts_supported_values_to_pixels():
    class ExistingLandmark:
        x = 0.25
        y = 0.5
        visibility = 0.8

    mapped = Landmark.from_value(
        {"x": 0.25, "y": 0.5, "confidence": 0.75}
    )
    tuple_value = Landmark.from_value((10.0, 20.0, 0.4), normalized=False)
    existing = Landmark.from_value(ExistingLandmark())

    assert mapped.visibility == 0.75
    assert mapped.coordinate_system == "normalized"
    assert mapped.to_pixel(200, 400) == (50.0, 200.0)
    assert tuple_value.coordinate_system == "pixel"
    assert tuple_value.to_pixel(200, 400) == (10.0, 20.0)
    assert existing.visibility == 0.8


@pytest.mark.parametrize(
    "value",
    [
        {"x": float("nan"), "y": 0.5},
        {"x": 0.5, "y": 0.5, "visibility": 1.1},
        (0.5,),
    ],
)
def test_landmark_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        Landmark.from_value(value)


def test_body_detections_normalises_keypoints_and_falls_back_per_side():
    detections = BodyDetections(
        keypoints={
            "left_heel": (10.0, 100.0, 0.9),
            "right_ankle": {"x": 20.0, "y": 101.0, "visibility": 0.8},
        },
        head_top=(15.0, 5.0),
        head_bottom=(15.0, 25.0),
        head_confidence=0.95,
    )

    assert detections.head_top == Landmark(15.0, 5.0)
    assert detections.head_bottom == Landmark(15.0, 25.0)
    assert detections.head_confidence == 0.95
    assert detections.head_bbox_coordinate_system == "pixel"
    assert detections.heel_landmarks() == (
        Landmark(10.0, 100.0, 0.9),
        Landmark(20.0, 101.0, 0.8),
    )
    assert BodyDetections().heel_landmarks() == ()


def test_quality_contracts_serialise_scores_and_recommendations():
    metrics = QualityMetrics(
        marker_visibility=1.0,
        pose_severity=0.1,
        blur_score=0.9,
        occlusion_score=0.8,
        model_agreement=0.7,
    )
    assessment = QualityAssessment(
        passed=True,
        metrics=metrics,
        recommendations=("Retake with better lighting",),
    )

    assert metrics.overall_score() == pytest.approx(0.86)
    assert metrics.level() == QualityLevel.HIGH
    assert metrics.overall_quality() == QualityLevel.HIGH
    assert assessment.to_dict() == {
        "passed": True,
        "metrics": {
            "marker_visibility": 1.0,
            "pose_severity": 0.1,
            "blur_score": 0.9,
            "occlusion_score": 0.8,
            "model_agreement": 0.7,
            "overall_score": 0.86,
            "overall_quality": "high",
        },
        "recommendations": ["Retake with better lighting"],
    }


@pytest.mark.parametrize(
    "field_name",
    [
        "marker_visibility",
        "pose_severity",
        "blur_score",
        "occlusion_score",
        "model_agreement",
    ],
)
def test_quality_metrics_reject_out_of_range_values(field_name):
    with pytest.raises(ValueError):
        QualityMetrics(**{field_name: 1.1})


def test_height_estimate_preserves_method_names_and_metadata():
    estimate = HeightEstimate(
        method=MeasurementMethod.GEOMETRIC,
        height_cm=172.35,
        confidence=0.875,
        notes="Head to heel",
        metadata={"scale_source": "aruco"},
    )

    assert estimate.to_dict() == {
        "method": "geometric",
        "height_cm": 172.35,
        "confidence": 0.875,
        "notes": "Head to heel",
        "metadata": {"scale_source": "aruco"},
    }


def test_height_estimate_supports_independent_estimator_methods():
    assert MeasurementMethod.SKELETON.value == "skeleton"
    assert MeasurementMethod.HEAD_BBOX.value == "head_bbox"


def test_empty_quality_metrics_are_not_high_confidence():
    metrics = QualityMetrics()

    assert metrics.overall_score() == pytest.approx(0.6)
    assert metrics.level() == QualityLevel.MODERATE


def test_height_estimate_rejects_non_json_metadata():
    with pytest.raises((TypeError, ValueError)):
        HeightEstimate(
            method=MeasurementMethod.GEOMETRIC,
            height_cm=172.0,
            confidence=0.8,
            metadata={"source": object()},
        )


def test_body_detections_validate_head_bbox_coordinate_system():
    detections = BodyDetections(
        head_bbox=(0.1, 0.2, 0.4, 0.8),
        head_bbox_coordinate_system="normalized",
    )

    assert detections.head_bbox_coordinate_system == "normalized"

    with pytest.raises(ValueError):
        BodyDetections(head_bbox_coordinate_system="camera")


def test_measurement_result_serialises_optional_estimates_and_diagnostics():
    quality = QualityAssessment(
        passed=True,
        metrics=QualityMetrics(marker_visibility=0.75, model_agreement=1.0),
    )
    result = MeasurementResult(
        estimated_height_cm=172.345,
        uncertainty_range=(169.1, 175.6),
        quality=quality,
        method_estimates=(
            HeightEstimate(
                method=MeasurementMethod.GEOMETRIC,
                height_cm=172.3,
                confidence=0.8,
            ),
        ),
        fusion_weights={"geometric": 1.0},
        measurements={"head_to_heel_cm": 172.3},
        diagnostics=("Only the geometric method was available",),
        warnings=("Hair endpoint inferred",),
    )

    serialised = result.to_dict()

    assert serialised["estimated_height_cm"] == 172.35
    assert serialised["uncertainty_range"] == {
        "lower_cm": 169.1,
        "upper_cm": 175.6,
        "total_range_cm": 6.5,
    }
    assert serialised["quality"]["metrics"]["overall_quality"] == "high"
    assert serialised["method_estimates"][0]["method"] == "geometric"
    assert serialised["fusion_weights"] == {"geometric": 1.0}
    assert serialised["measurements"] == {"head_to_heel_cm": 172.3}
    assert serialised["diagnostics"] == ["Only the geometric method was available"]
    assert serialised["warnings"] == ["Hair endpoint inferred"]

    minimal = MeasurementResult(
        estimated_height_cm=170.0,
        uncertainty_range=(170.0, 170.0),
        quality=quality,
    )
    assert minimal.to_dict()["method_estimates"] == []
