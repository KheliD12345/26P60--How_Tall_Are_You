from __future__ import annotations

from math import isfinite, sqrt
from statistics import NormalDist, mean
from typing import Mapping, Sequence

from .advanced_models import (
    HeightEstimate,
    MeasurementMethod,
    MeasurementResult,
    QualityAssessment,
    QualityLevel,
    QualityMetrics,
)


DEFAULT_BASE_WEIGHTS = {
    MeasurementMethod.GEOMETRIC.value: 0.5,
    MeasurementMethod.HEAD_BBOX.value: 0.2,
    MeasurementMethod.ANTHROPOMETRIC.value: 0.3,
}


def _quality_factor(quality: QualityAssessment) -> float:
    return {
        QualityLevel.HIGH: 0.8,
        QualityLevel.MODERATE: 1.0,
        QualityLevel.LOW: 1.3,
        QualityLevel.UNUSABLE: 2.0,
    }[quality.quality_level]


class UncertaintyEstimator:
    """Estimate a conservative interval from estimate spread and quality."""

    def __init__(self, *, confidence_level: float = 0.95) -> None:
        if not isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence level must be between zero and one")
        self.confidence_level = confidence_level

    def interval(
        self,
        *,
        fused_height_cm: float,
        estimates: Sequence[HeightEstimate],
        quality: QualityAssessment,
        weights: Mapping[str, float] | None = None,
    ) -> tuple[float, float]:
        if not isfinite(fused_height_cm) or fused_height_cm <= 0.0:
            raise ValueError("fused height must be finite and positive")
        if not estimates:
            raise ValueError("at least one estimate is required")

        values = [estimate.height_cm for estimate in estimates]
        average_confidence = mean(estimate.confidence for estimate in estimates)
        if len(values) == 1:
            standard_error = fused_height_cm * 0.02
        else:
            if weights is None:
                squared_spread = mean(
                    (value - fused_height_cm) ** 2 for value in values
                )
            else:
                squared_spread = sum(
                    weights[estimate.method.value]
                    * (estimate.height_cm - fused_height_cm) ** 2
                    for estimate in estimates
                )
            spread = sqrt(max(0.0, squared_spread))
            confidence_error = fused_height_cm * (1.0 - average_confidence) * 0.05
            standard_error = sqrt(spread**2 + confidence_error**2)

        z_value = NormalDist().inv_cdf(
            0.5 + self.confidence_level / 2.0
        )
        margin = z_value * standard_error * _quality_factor(quality)
        margin = max(margin, fused_height_cm * 0.005)
        lower = max(0.0, fused_height_cm - margin)
        upper = fused_height_cm + margin
        return lower, upper


class MeasurementFusionEngine:
    """Combine independent height estimates without fabricating measurements."""

    def __init__(
        self,
        *,
        base_weights: Mapping[str, float] | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
    ) -> None:
        configured_weights = dict(DEFAULT_BASE_WEIGHTS)
        if base_weights is not None:
            configured_weights.update(base_weights)
        if not configured_weights:
            raise ValueError("base weights cannot be empty")
        for method, weight in configured_weights.items():
            if not isinstance(method, str) or not method:
                raise ValueError("base weight method names must be non-empty")
            if not isfinite(float(weight)) or float(weight) < 0.0:
                raise ValueError("base weights must be finite and non-negative")
        if sum(configured_weights.values()) <= 0.0:
            raise ValueError("base weights must contain a positive value")
        self.base_weights = {
            method: float(weight) / sum(configured_weights.values())
            for method, weight in configured_weights.items()
        }
        self.uncertainty_estimator = (
            uncertainty_estimator or UncertaintyEstimator()
        )

    def fuse(
        self,
        estimates: Sequence[HeightEstimate],
        *,
        quality: QualityAssessment | None = None,
    ) -> MeasurementResult:
        valid_estimates = tuple(estimates)
        if not valid_estimates:
            raise ValueError("at least one estimate is required")
        if any(not isinstance(estimate, HeightEstimate) for estimate in valid_estimates):
            raise TypeError("estimates must be HeightEstimate instances")

        methods = [estimate.method.value for estimate in valid_estimates]
        if len(methods) != len(set(methods)):
            raise ValueError("duplicate estimate methods are not supported")
        if MeasurementMethod.SMPL_BASED.value in methods:
            raise ValueError("SMPL estimates require a real fitted estimator")

        raw_weights: dict[str, float] = {}
        for estimate in valid_estimates:
            base_weight = self.base_weights.get(estimate.method.value)
            if base_weight is None:
                raise ValueError(f"no base weight configured for {estimate.method.value}")
            raw_weights[estimate.method.value] = base_weight * estimate.confidence
        total_weight = sum(raw_weights.values())
        if total_weight <= 0.0:
            raise ValueError("estimate confidence produces zero total weight")
        weights = {
            method: weight / total_weight
            for method, weight in raw_weights.items()
        }

        fused_height = sum(
            estimate.height_cm * weights[estimate.method.value]
            for estimate in valid_estimates
        )
        if not isfinite(fused_height) or fused_height <= 0.0:
            raise ValueError("fused height must be finite and positive")

        agreement = self._agreement(valid_estimates, fused_height, weights)
        source_quality = quality or QualityAssessment(
            passed=True,
            metrics=QualityMetrics(),
        )
        updated_quality = QualityAssessment(
            passed=source_quality.passed,
            metrics=QualityMetrics(
                marker_visibility=source_quality.metrics.marker_visibility,
                pose_severity=source_quality.metrics.pose_severity,
                blur_score=source_quality.metrics.blur_score,
                occlusion_score=source_quality.metrics.occlusion_score,
                model_agreement=agreement,
            ),
            recommendations=source_quality.recommendations,
        )
        uncertainty_range = self.uncertainty_estimator.interval(
            fused_height_cm=fused_height,
            estimates=valid_estimates,
            quality=updated_quality,
            weights=weights,
        )
        methods_used = ", ".join(methods)
        diagnostics = (
            f"Fused {len(valid_estimates)} estimate(s) from: {methods_used}",
            f"Model agreement: {agreement:.3f}",
            f"Estimate spread: {self._spread(valid_estimates, fused_height):.3f} cm",
        )
        measurements = {
            "estimate_count": float(len(valid_estimates)),
            "fused_height_cm": fused_height,
            "model_agreement": agreement,
            "estimate_spread_cm": self._spread(valid_estimates, fused_height),
        }
        return MeasurementResult(
            estimated_height_cm=fused_height,
            uncertainty_range=uncertainty_range,
            quality=updated_quality,
            method_estimates=valid_estimates,
            fusion_weights=weights,
            measurements=measurements,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _spread(
        estimates: Sequence[HeightEstimate],
        fused_height: float,
    ) -> float:
        return sqrt(
            mean((estimate.height_cm - fused_height) ** 2 for estimate in estimates)
        )

    @classmethod
    def _agreement(
        cls,
        estimates: Sequence[HeightEstimate],
        fused_height: float,
        weights: Mapping[str, float],
    ) -> float:
        if len(estimates) == 1:
            return 0.25
        relative_spread = cls._spread(estimates, fused_height) / fused_height
        pairwise_differences = [
            abs(left.height_cm - right.height_cm) / mean(
                (left.height_cm, right.height_cm)
            )
            for index, left in enumerate(estimates)
            for right in estimates[index + 1 :]
        ]
        pairwise_difference = mean(pairwise_differences)
        confidence_adjustment = sum(
            weights[estimate.method.value] * estimate.confidence
            for estimate in estimates
        )
        return max(
            0.0,
            min(
                1.0,
                1.0
                - (0.5 * relative_spread + 0.5 * pairwise_difference)
                * (1.0 + (1.0 - confidence_adjustment)),
            ),
        )


def fuse_height_estimates(
    estimates: Sequence[HeightEstimate],
    *,
    quality: QualityAssessment | None = None,
    base_weights: Mapping[str, float] | None = None,
) -> MeasurementResult:
    return build_measurement_result(
        estimates,
        quality=quality,
        base_weights=base_weights,
    )


def build_measurement_result(
    estimates: Sequence[HeightEstimate],
    *,
    quality: QualityAssessment | None = None,
    base_weights: Mapping[str, float] | None = None,
    confidence_level: float = 0.95,
) -> MeasurementResult:
    """Build the stable result contract from already-produced estimates."""
    return MeasurementFusionEngine(
        base_weights=base_weights,
        uncertainty_estimator=UncertaintyEstimator(
            confidence_level=confidence_level,
        ),
    ).fuse(estimates, quality=quality)