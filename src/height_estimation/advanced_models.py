from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping


class QualityLevel(Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNUSABLE = "unusable"


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float = 1.0
    normalized: bool = True
    coordinate_system: str | None = None

    def __post_init__(self) -> None:
        try:
            coordinates = (float(self.x), float(self.y))
            visibility = float(self.visibility)
        except (TypeError, ValueError) as error:
            raise ValueError("landmark values must be numeric") from error

        coordinate_system = self.coordinate_system
        if coordinate_system is None:
            coordinate_system = "normalized" if self.normalized else "pixel"
        if coordinate_system not in {"normalized", "pixel"}:
            raise ValueError(
                "landmark coordinate_system must be normalized or pixel"
            )

        if not all(isfinite(value) for value in (*coordinates, visibility)):
            raise ValueError("landmark values must be finite")
        if not 0.0 <= visibility <= 1.0:
            raise ValueError("landmark visibility must be between zero and one")

        object.__setattr__(self, "x", coordinates[0])
        object.__setattr__(self, "y", coordinates[1])
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "coordinate_system", coordinate_system)
        object.__setattr__(
            self,
            "normalized",
            coordinate_system == "normalized",
        )

    @property
    def confidence(self) -> float:
        return self.visibility

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        normalized: bool = True,
        coordinate_system: str | None = None,
    ) -> "Landmark":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            try:
                x = value["x"]
                y = value["y"]
            except KeyError as error:
                raise ValueError(
                    "landmark values require x and y coordinates"
                ) from error
            visibility = value.get(
                "visibility",
                value.get("confidence", value.get("score", 1.0)),
            )
            source_normalized = bool(value.get("normalized", normalized))
            source_system = value.get("coordinate_system", coordinate_system)
            return cls(
                float(x),
                float(y),
                float(visibility),
                source_normalized,
                source_system,
            )
        if isinstance(value, (tuple, list)):
            if len(value) < 2:
                raise ValueError("landmark values require x and y coordinates")
            visibility = float(value[2]) if len(value) >= 3 else 1.0
            return cls(
                float(value[0]),
                float(value[1]),
                visibility,
                normalized,
                coordinate_system,
            )

        try:
            x = getattr(value, "x")
            y = getattr(value, "y")
        except AttributeError as error:
            raise ValueError(
                "landmark values require x and y coordinates"
            ) from error
        visibility = getattr(
            value,
            "visibility",
            getattr(value, "confidence", getattr(value, "score", 1.0)),
        )
        source_normalized = bool(getattr(value, "normalized", normalized))
        source_system = getattr(value, "coordinate_system", coordinate_system)
        return cls(
            float(x),
            float(y),
            float(visibility),
            source_normalized,
            source_system,
        )

    def to_pixel(
        self,
        image_width: int | float,
        image_height: int | float,
    ) -> tuple[float, float]:
        try:
            width = float(image_width)
            height = float(image_height)
        except (TypeError, ValueError) as error:
            raise ValueError("image dimensions must be finite and positive") from error
        if not isfinite(width) or not isfinite(height) or width <= 0 or height <= 0:
            raise ValueError("image dimensions must be finite and positive")
        if self.coordinate_system == "normalized":
            return self.x * width, self.y * height
        return self.x, self.y


@dataclass
class BodyDetections:
    keypoints: dict[str, Landmark] = field(default_factory=dict)
    head_top: Landmark | None = None
    head_bottom: Landmark | None = None
    head_bbox: tuple[float, float, float, float] | None = None
    head_confidence: float = 0.0
    hair_top: Landmark | None = None
    hair_bottom: Landmark | None = None
    hand_lengths: tuple[tuple[Landmark, Landmark], ...] = ()
    segmentation_mask: Any | None = None

    def heel_points(self) -> tuple[Landmark, ...]:
        heels = tuple(
            self.keypoints[name]
            for name in ("left_heel", "right_heel")
            if name in self.keypoints
        )
        if heels:
            return heels
        return tuple(
            self.keypoints[name]
            for name in ("left_ankle", "right_ankle")
            if name in self.keypoints
        )


@dataclass(frozen=True)
class QualityMetrics:
    marker_visibility: float = 0.0
    pose_severity: float = 0.0
    blur_score: float = 1.0
    occlusion_score: float = 1.0
    model_agreement: float = 1.0

    def overall_score(self) -> float:
        return (
            self.marker_visibility
            + (1.0 - self.pose_severity)
            + self.blur_score
            + self.occlusion_score
            + self.model_agreement
        ) / 5.0

    def level(self) -> QualityLevel:
        score = self.overall_score()
        if score >= 0.8:
            return QualityLevel.HIGH
        if score >= 0.6:
            return QualityLevel.MODERATE
        if score >= 0.4:
            return QualityLevel.LOW
        return QualityLevel.UNUSABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_visibility": round(self.marker_visibility, 3),
            "pose_severity": round(self.pose_severity, 3),
            "blur_score": round(self.blur_score, 3),
            "occlusion_score": round(self.occlusion_score, 3),
            "model_agreement": round(self.model_agreement, 3),
            "overall_score": round(self.overall_score(), 3),
            "overall_quality": self.level().value,
        }


@dataclass(frozen=True)
class QualityAssessment:
    passed: bool
    metrics: QualityMetrics
    recommendations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics.to_dict(),
            "recommendations": list(self.recommendations),
        }


@dataclass(frozen=True)
class HeightEstimate:
    method: str
    height_cm: float
    confidence: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "height_cm": round(self.height_cm, 2),
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AdvancedMeasurementResult:
    estimated_height_cm: float
    uncertainty_range: tuple[float, float]
    quality: QualityAssessment
    method_estimates: tuple[HeightEstimate, ...]
    fusion_weights: dict[str, float]
    measurements: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        lower, upper = self.uncertainty_range
        return {
            "estimated_height_cm": round(self.estimated_height_cm, 2),
            "uncertainty_range": {
                "lower_cm": round(lower, 2),
                "upper_cm": round(upper, 2),
                "total_range_cm": round(upper - lower, 2),
            },
            "quality": self.quality.to_dict(),
            "method_estimates": [
                estimate.to_dict() for estimate in self.method_estimates
            ],
            "fusion_weights": {
                name: round(weight, 3)
                for name, weight in self.fusion_weights.items()
            },
            "measurements": {
                name: round(value, 2)
                for name, value in self.measurements.items()
            },
        }