import json

import pytest

from height_estimation.advanced_models import (
    BodyDetections,
    HeightEstimate,
    MeasurementMethod,
    MeasurementResult,
    QualityAssessment,
    QualityLevel,
    QualityMetrics,
)
from height_estimation.estimators import estimate_independent_heights
from height_estimation.fusion import (
    MeasurementFusionEngine,
    UncertaintyEstimator,
    build_measurement_result,
)


def make_quality(
    *,
    marker_visibility: float = 1.0,
    blur_score: float = 1.0,
    occlusion_score: float = 1.0,
) -> QualityAssessment:
    return QualityAssessment(
        passed=True,
        metrics=QualityMetrics(
            marker_visibility=marker_visibility,
            blur_score=blur_score,
            occlusion_score=occlusion_score,
        ),
    )


def estimate(
    method: MeasurementMethod,
    height_cm: float,
    confidence: float,
) -> HeightEstimate:
    return HeightEstimate(
        method=method,
        height_cm=height_cm,
        confidence=confidence,
    )


def test_empty_estimates_are_rejected():
    with pytest.raises(ValueError, match="estimate"):
        MeasurementFusionEngine().fuse(())


def test_single_estimate_is_fused_with_conservative_agreement():
    result = MeasurementFusionEngine().fuse(
        [estimate(MeasurementMethod.GEOMETRIC, 172.0, 0.9)],
        quality=make_quality(),
    )

    assert isinstance(result, MeasurementResult)
    assert result.estimated_height_cm == 172.0
    assert result.fusion_weights == {"geometric": 1.0}
    assert 0.0 <= result.quality.metrics.model_agreement < 1.0
    assert result.uncertainty_range[0] < 172.0 < result.uncertainty_range[1]


def test_multiple_estimates_use_confidence_weighted_fusion():
    result = MeasurementFusionEngine().fuse(
        [
            estimate(MeasurementMethod.GEOMETRIC, 170.0, 1.0),
            estimate(MeasurementMethod.HEAD_BBOX, 180.0, 0.5),
        ],
        quality=make_quality(),
    )

    assert result.estimated_height_cm == pytest.approx(171.666666667)
    assert result.fusion_weights["geometric"] > result.fusion_weights["head_bbox"]
    assert sum(result.fusion_weights.values()) == pytest.approx(1.0)
    assert result.quality.metrics.model_agreement < 1.0


def test_configurable_base_weights_are_applied_only_to_present_methods():
    result = MeasurementFusionEngine(
        base_weights={"geometric": 0.1, "head_bbox": 0.9}
    ).fuse(
        [
            estimate(MeasurementMethod.GEOMETRIC, 170.0, 1.0),
            estimate(MeasurementMethod.HEAD_BBOX, 180.0, 1.0),
            estimate(MeasurementMethod.ANTHROPOMETRIC, 175.0, 1.0),
        ]
    )

    assert set(result.fusion_weights) == {"geometric", "head_bbox", "anthropometric"}
    assert result.fusion_weights["head_bbox"] > result.fusion_weights["geometric"]
    assert result.fusion_weights["anthropometric"] > 0.0


def test_duplicate_methods_are_rejected():
    estimates = [
        estimate(MeasurementMethod.GEOMETRIC, 170.0, 0.8),
        estimate(MeasurementMethod.GEOMETRIC, 171.0, 0.8),
    ]

    with pytest.raises(ValueError, match="duplicate"):
        MeasurementFusionEngine().fuse(estimates)


def test_zero_total_confidence_is_rejected_without_division_by_zero():
    estimates = [
        estimate(MeasurementMethod.GEOMETRIC, 170.0, 0.0),
        estimate(MeasurementMethod.HEAD_BBOX, 171.0, 0.0),
    ]

    with pytest.raises(ValueError, match="weight|confidence"):
        MeasurementFusionEngine().fuse(estimates)


def test_quality_and_estimate_spread_expand_uncertainty():
    estimates = [
        estimate(MeasurementMethod.GEOMETRIC, 160.0, 0.8),
        estimate(MeasurementMethod.HEAD_BBOX, 180.0, 0.8),
    ]
    good = MeasurementFusionEngine().fuse(estimates, quality=make_quality())
    poor = MeasurementFusionEngine().fuse(
        estimates,
        quality=make_quality(
            marker_visibility=0.2,
            blur_score=0.2,
            occlusion_score=0.2,
        ),
    )

    assert good.uncertainty_range[0] <= good.estimated_height_cm
    assert good.estimated_height_cm <= good.uncertainty_range[1]
    assert poor.uncertainty_range[0] < good.uncertainty_range[0]
    assert poor.uncertainty_range[1] > good.uncertainty_range[1]


def test_uncertainty_estimator_supports_configurable_confidence_level():
    estimator = UncertaintyEstimator(confidence_level=0.99)
    interval = estimator.interval(
        fused_height_cm=170.0,
        estimates=(
            estimate(MeasurementMethod.GEOMETRIC, 169.0, 0.9),
            estimate(MeasurementMethod.HEAD_BBOX, 171.0, 0.9),
        ),
        quality=make_quality(),
    )

    assert interval[0] < 170.0 < interval[1]
    assert all(value >= 0.0 for value in interval)


def test_uncertainty_interval_grows_monotonically_with_confidence_level():
    estimates = (
        estimate(MeasurementMethod.GEOMETRIC, 165.0, 0.9),
        estimate(MeasurementMethod.HEAD_BBOX, 175.0, 0.9),
    )
    quality = make_quality()

    ninety_percent = UncertaintyEstimator(confidence_level=0.90).interval(
        fused_height_cm=170.0,
        estimates=estimates,
        quality=quality,
    )
    ninety_five_percent = UncertaintyEstimator(confidence_level=0.95).interval(
        fused_height_cm=170.0,
        estimates=estimates,
        quality=quality,
    )
    ninety_nine_percent = UncertaintyEstimator(confidence_level=0.99).interval(
        fused_height_cm=170.0,
        estimates=estimates,
        quality=quality,
    )

    assert ninety_five_percent[0] < ninety_percent[0]
    assert ninety_five_percent[1] > ninety_percent[1]
    assert ninety_nine_percent[0] < ninety_five_percent[0]
    assert ninety_nine_percent[1] > ninety_five_percent[1]


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1])
def test_uncertainty_estimator_rejects_invalid_confidence_levels(confidence_level):
    with pytest.raises(ValueError, match="confidence level"):
        UncertaintyEstimator(confidence_level=confidence_level)


def test_fusion_result_is_json_safe_and_preserves_methods():
    result = MeasurementFusionEngine().fuse(
        [
            estimate(MeasurementMethod.GEOMETRIC, 171.0, 0.9),
            estimate(MeasurementMethod.ANTHROPOMETRIC, 172.0, 0.8),
        ],
        quality=make_quality(),
    )

    payload = result.to_dict()

    assert json.loads(result.to_json()) == payload
    assert [item["method"] for item in payload["method_estimates"]] == [
        "geometric",
        "anthropometric",
    ]
    assert payload["diagnostics"]
    assert payload["measurements"]


def test_unsupported_methods_are_rejected_by_the_existing_estimate_contract():
    with pytest.raises(ValueError, match="supported"):
        HeightEstimate(method="unknown", height_cm=170.0, confidence=0.8)


def test_smpl_estimate_is_not_fabricated_by_fusion():
    with pytest.raises(ValueError, match="SMPL|smpl|unsupported"):
        MeasurementFusionEngine().fuse(
            [estimate(MeasurementMethod.SMPL_BASED, 170.0, 0.9)]
        )


def test_build_measurement_result_integrates_independent_estimator_outputs():
    estimates = estimate_independent_heights(
        BodyDetections(
            head_top=(0.5, 0.1),
            head_bottom=(0.5, 0.23),
            head_bbox=(0.4, 0.1, 0.6, 0.3),
            head_confidence=0.9,
            keypoints={"left_heel": (0.5, 0.9)},
        ),
        cm_per_pixel=100.0,
        image_size=(100, 100),
    )

    result = build_measurement_result(estimates, quality=make_quality())

    assert result.method_estimates == estimates
    assert result.estimated_height_cm > 0.0
    assert result.uncertainty_range[0] <= result.estimated_height_cm
    assert result.estimated_height_cm <= result.uncertainty_range[1]
    assert json.loads(result.to_json()) == result.to_dict()