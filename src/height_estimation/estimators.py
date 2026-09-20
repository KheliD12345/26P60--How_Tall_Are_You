from math import isfinite
from typing import Any

from .advanced_models import (
    BodyDetections,
    HeightEstimate,
    Landmark,
    MeasurementMethod,
    QualityAssessment,
    QualityMetrics,
)


def _quality_score(
    quality: QualityAssessment | QualityMetrics | None,
) -> float:
    if quality is None:
        return 0.5
    return quality.overall_score


def _pixel_point(
    landmark: Landmark,
    image_size: tuple[int, int] | None,
) -> tuple[float, float]:
    if landmark.coordinate_system == "normalized":
        if image_size is None or len(image_size) != 2:
            raise ValueError(
                "image_size is required for normalized body landmarks"
            )
        return landmark.to_pixel(image_size[1], image_size[0])
    return landmark.x, landmark.y


def _positive_height(value: float) -> float | None:
    return value if isfinite(value) and value > 0.0 else None


class GeometricHeightEstimator:
    """Estimate stature from calibrated head and lower-body endpoints."""

    def estimate(
        self,
        detections: BodyDetections,
        *,
        cm_per_pixel: float | None = None,
        image_size: tuple[int, int] | None = None,
        metric_endpoints: tuple[
            tuple[float, float], tuple[float, float]
        ] | None = None,
        quality: QualityAssessment | QualityMetrics | None = None,
    ) -> HeightEstimate | None:
        if metric_endpoints is not None:
            if not all(
                isfinite(value)
                for point in metric_endpoints
                for value in point
            ):
                return None
            height_cm = abs(metric_endpoints[1][1] - metric_endpoints[0][1])
            height_cm = _positive_height(height_cm)
            if height_cm is None:
                return None
            return HeightEstimate(
                method=MeasurementMethod.GEOMETRIC,
                height_cm=height_cm,
                confidence=min(1.0, _quality_score(quality)),
                notes="Direct head-to-feet measurement on the metric plane.",
                metadata={
                    "coordinate_system": "metric_plane",
                    "input_endpoints": [
                        list(metric_endpoints[0]),
                        list(metric_endpoints[1]),
                    ],
                },
            )

        if not isfinite(cm_per_pixel or float("nan")) or (cm_per_pixel or 0) <= 0:
            return None

        head_value = detections.head_top or detections.keypoints.get("head_top")
        if head_value is None:
            return None

        head = Landmark.from_value(head_value)
        heels = detections.heel_landmarks()
        if not heels:
            return None
        if any(landmark.coordinate_system != head.coordinate_system for landmark in heels):
            return None

        try:
            head_point = _pixel_point(head, image_size)
            heel_points = [_pixel_point(landmark, image_size) for landmark in heels]
        except ValueError:
            return None

        heel = max(heel_points, key=lambda point: point[1])
        height_px = abs(heel[1] - head_point[1])
        height_cm = _positive_height(height_px * cm_per_pixel)
        if height_cm is None:
            return None

        endpoint_visibility = (head.visibility + max(
            landmark.visibility for landmark in heels
        )) / 2.0
        confidence = min(1.0, endpoint_visibility * _quality_score(quality))
        fallback = "heel" if any(
            name.endswith("_heel") for name in detections.keypoints
        ) else "ankle"
        coordinate_system = head.coordinate_system
        return HeightEstimate(
            method=MeasurementMethod.GEOMETRIC,
            height_cm=height_cm,
            confidence=confidence,
            notes=f"Direct head-to-{fallback} measurement.",
            metadata={
                "coordinate_system": coordinate_system,
                "cm_per_pixel": cm_per_pixel,
                "input_landmarks": ["head_top", f"{fallback}_landmarks"],
                "endpoint_visibility": [
                    head.visibility,
                    max(landmark.visibility for landmark in heels),
                ],
                "fallback_used": fallback == "ankle",
            },
        )


class HeadBoundingBoxHeightEstimator:
    """Placeholder until the independent head-box estimator is implemented."""

    def estimate(self, detections: BodyDetections, **kwargs: Any) -> None:
        return None


class AnthropometricHeightEstimator:
    """Placeholder until the independent anthropometric estimator is implemented."""

    def estimate(self, detections: BodyDetections, **kwargs: Any) -> None:
        return None