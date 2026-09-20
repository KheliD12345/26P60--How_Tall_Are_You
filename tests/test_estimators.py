import json

import pytest

from height_estimation.advanced_models import (
    BodyDetections,
    HeightEstimate,
    Landmark,
    MeasurementMethod,
    QualityAssessment,
    QualityMetrics,
)
from height_estimation.estimators import (
    AnthropometricHeightEstimator,
    GeometricHeightEstimator,
    HeadBoundingBoxHeightEstimator,
)


def high_quality() -> QualityAssessment:
    return QualityAssessment(
        passed=True,
        metrics=QualityMetrics(
            marker_visibility=1.0,
            pose_severity=0.0,
            blur_score=1.0,
            occlusion_score=1.0,
            model_agreement=1.0,
        ),
    )


def test_geometric_estimator_uses_head_and_heel_pixels():
    detections = BodyDetections(
        head_top=Landmark(100.0, 20.0, normalized=False),
        keypoints={"left_heel": Landmark(101.0, 220.0, normalized=False)},
    )

    estimate = GeometricHeightEstimator().estimate(
        detections,
        cm_per_pixel=0.8,
        quality=high_quality(),
    )

    assert estimate is not None
    assert estimate.method is MeasurementMethod.GEOMETRIC
    assert estimate.height_cm == pytest.approx(160.0)
    assert 0.0 <= estimate.confidence <= 1.0
    assert estimate.metadata["coordinate_system"] == "pixel"


def test_geometric_estimator_falls_back_from_heels_to_ankles():
    detections = BodyDetections(
        head_top=Landmark(0.5, 0.1),
        keypoints={"right_ankle": Landmark(0.5, 0.9)},
    )

    estimate = GeometricHeightEstimator().estimate(
        detections,
        cm_per_pixel=2.0,
        image_size=(100, 100),
    )

    assert estimate is not None
    assert estimate.height_cm == pytest.approx(160.0)
    assert "ankle" in estimate.notes.lower()


def test_geometric_estimator_accepts_metric_plane_endpoints():
    detections = BodyDetections()

    estimate = GeometricHeightEstimator().estimate(
        detections,
        metric_endpoints=((20.0, 180.0), (20.0, 0.0)),
    )

    assert estimate is not None
    assert estimate.height_cm == pytest.approx(180.0)
    assert estimate.metadata["coordinate_system"] == "metric_plane"


def test_geometric_estimator_returns_none_without_complete_endpoints():
    assert GeometricHeightEstimator().estimate(
        BodyDetections(head_top=Landmark(0.5, 0.1)),
        cm_per_pixel=100.0,
    ) is None


def test_head_bbox_estimator_supports_normalized_bbox():
    detections = BodyDetections(
        head_bbox=(0.4, 0.1, 0.6, 0.3),
        head_confidence=0.9,
        head_bbox_coordinate_system="normalized",
    )

    estimate = HeadBoundingBoxHeightEstimator().estimate(
        detections,
        cm_per_pixel=1.0,
        image_size=(200, 100),
    )

    assert estimate is not None
    assert estimate.method is MeasurementMethod.HEAD_BBOX
    assert estimate.height_cm == pytest.approx(40.0)
    assert estimate.metadata["approximation"] is True
    assert 0.0 <= estimate.confidence <= 1.0


def test_head_bbox_estimator_prefers_explicit_head_landmarks():
    detections = BodyDetections(
        head_bbox=(40.0, 20.0, 80.0, 70.0),
        head_top=Landmark(60.0, 15.0, normalized=False),
        head_bottom=Landmark(60.0, 75.0, normalized=False),
        head_confidence=0.8,
    )

    estimate = HeadBoundingBoxHeightEstimator().estimate(
        detections,
        cm_per_pixel=0.5,
    )

    assert estimate is not None
    assert estimate.height_cm == pytest.approx(30.0)
    assert estimate.metadata["approximation"] is False


def test_head_bbox_estimator_returns_none_without_valid_bbox_or_landmarks():
    assert HeadBoundingBoxHeightEstimator().estimate(
        BodyDetections(head_confidence=0.9),
        cm_per_pixel=1.0,
    ) is None


def test_anthropometric_estimator_uses_head_ratio():
    detections = BodyDetections(
        head_top=Landmark(0.5, 0.1),
        head_bottom=Landmark(0.5, 0.23),
    )

    estimate = AnthropometricHeightEstimator().estimate(
        detections,
        cm_per_pixel=1000.0,
        image_size=(100, 100),
        quality=high_quality(),
    )

    assert estimate is not None
    assert estimate.method is MeasurementMethod.ANTHROPOMETRIC
    assert estimate.height_cm == pytest.approx(100.0)
    assert "head" in estimate.metadata["measurements_used"]
    assert estimate.confidence < 1.0


def test_anthropometric_estimator_uses_available_arm_segment():
    detections = BodyDetections(
        keypoints={
            "left_shoulder": Landmark(0.2, 0.2),
            "left_wrist": Landmark(0.2, 0.66),
        }
    )

    estimate = AnthropometricHeightEstimator().estimate(
        detections,
        cm_per_pixel=1000.0,
        image_size=(100, 100),
    )

    assert estimate is not None
    assert estimate.height_cm == pytest.approx(100.0)
    assert estimate.metadata["measurements_used"] == ["left_arm"]


def test_anthropometric_estimator_returns_none_without_segments():
    assert AnthropometricHeightEstimator().estimate(
        BodyDetections(),
        cm_per_pixel=1.0,
    ) is None


@pytest.mark.parametrize(
    "estimator, detections, kwargs",
    [
        (
            GeometricHeightEstimator(),
            BodyDetections(
                head_top=Landmark(0.5, 0.5),
                keypoints={"left_heel": Landmark(0.5, 0.5)},
            ),
            {"cm_per_pixel": 100.0},
        ),
        (
            HeadBoundingBoxHeightEstimator(),
            BodyDetections(
                head_bbox=(0.4, 0.2, 0.4, 0.2),
                head_confidence=0.9,
                head_bbox_coordinate_system="normalized",
            ),
            {"cm_per_pixel": 100.0, "image_size": (100, 100)},
        ),
    ],
)
def test_estimators_reject_non_positive_or_degenerate_measurements(
    estimator,
    detections,
    kwargs,
):
    assert estimator.estimate(detections, **kwargs) is None


def test_estimators_produce_json_safe_height_estimates():
    estimate = HeightEstimate(
        method=MeasurementMethod.GEOMETRIC,
        height_cm=170.0,
        confidence=0.8,
        metadata={"coordinates": (1.0, 2.0), "fallbacks": ["ankle"]},
    )

    assert json.loads(json.dumps(estimate.to_dict())) == estimate.to_dict()
