from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping


class MeasurementMethod(str, Enum):
    GEOMETRIC = "geometric"
    SMPL_BASED = "smpl_based"
    ANTHROPOMETRIC = "anthropometric"
    FUSION = "fusion"


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


LandmarkValue = (
    Landmark
    | Mapping[str, Any]
    | tuple[float, ...]
    | list[float]
)


def _bounded_score(value: float, field_name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric") from error
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{field_name} must be between zero and one")
    return score


@dataclass
class BodyDetections:
    keypoints: Mapping[str, LandmarkValue] = field(default_factory=dict)
    head_top: LandmarkValue | None = None
    head_bottom: LandmarkValue | None = None
    head_bbox: tuple[float, float, float, float] | None = None
    head_confidence: float = 0.0
    hair_top: LandmarkValue | None = None
    hair_bottom: LandmarkValue | None = None
    hand_lengths: tuple[tuple[LandmarkValue, LandmarkValue], ...] = ()
    segmentation_mask: Any | None = None

    def __post_init__(self) -> None:
        self.keypoints = {
            str(name): Landmark.from_value(value)
            for name, value in self.keypoints.items()
        }
        self.head_top = self._landmark_or_none(self.head_top)
        self.head_bottom = self._landmark_or_none(self.head_bottom)
        self.hair_top = self._landmark_or_none(self.hair_top)
        self.hair_bottom = self._landmark_or_none(self.hair_bottom)
        self.head_confidence = _bounded_score(
            self.head_confidence,
            "head confidence",
        )
        if self.head_bbox is not None:
            if len(self.head_bbox) != 4:
                raise ValueError("head bounding box must contain four values")
            try:
                bbox = tuple(float(value) for value in self.head_bbox)
            except (TypeError, ValueError) as error:
                raise ValueError("head bounding box values must be numeric") from error
            if not all(isfinite(value) for value in bbox):
                raise ValueError("head bounding box values must be finite")
            self.head_bbox = bbox
        self.hand_lengths = tuple(
            (
                Landmark.from_value(first),
                Landmark.from_value(second),
            )
            for first, second in self.hand_lengths
        )

    @staticmethod
    def _landmark_or_none(value: LandmarkValue | None) -> Landmark | None:
        return None if value is None else Landmark.from_value(value)

    def heel_landmarks(self) -> tuple[Landmark, ...]:
        heels: list[Landmark] = []
        for side in ("left", "right"):
            heel = self.keypoints.get(f"{side}_heel")
            ankle = self.keypoints.get(f"{side}_ankle")
            if heel is not None:
                heels.append(heel)
            elif ankle is not None:
                heels.append(ankle)
        return tuple(heels)

    def heel_points(self) -> tuple[Landmark, ...]:
        return self.heel_landmarks()


@dataclass(frozen=True)
class QualityMetrics:
    marker_visibility: float = 0.0
    pose_severity: float = 0.0
    blur_score: float = 1.0
    occlusion_score: float = 1.0
    model_agreement: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "marker_visibility",
            "pose_severity",
            "blur_score",
            "occlusion_score",
            "model_agreement",
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_score(getattr(self, field_name), field_name),
            )

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

    def overall_quality(self) -> QualityLevel:
        return self.level()

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

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("quality assessment passed must be a boolean")
        if not isinstance(self.metrics, QualityMetrics):
            raise TypeError("quality assessment metrics must be QualityMetrics")
        recommendations = tuple(self.recommendations)
        if any(
            not isinstance(recommendation, str) or not recommendation.strip()
            for recommendation in recommendations
        ):
            raise ValueError("quality recommendations must be non-empty strings")
        object.__setattr__(self, "recommendations", recommendations)

    @property
    def overall_score(self) -> float:
        return self.metrics.overall_score()

    @property
    def quality_level(self) -> QualityLevel:
        return self.metrics.level()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics.to_dict(),
            "recommendations": list(self.recommendations),
        }


@dataclass(frozen=True)
class HeightEstimate:
    method: MeasurementMethod | str
    height_cm: float
    confidence: float
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            method = (
                self.method
                if isinstance(self.method, MeasurementMethod)
                else MeasurementMethod(str(self.method))
            )
        except ValueError as error:
            raise ValueError(
                "height estimate method must be a supported measurement method"
            ) from error

        try:
            height_cm = float(self.height_cm)
        except (TypeError, ValueError) as error:
            raise ValueError("height estimate must be numeric") from error
        if not isfinite(height_cm) or height_cm <= 0.0:
            raise ValueError("height estimate must be finite and positive")

        if not isinstance(self.notes, str):
            raise ValueError("height estimate notes must be a string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("height estimate metadata must be a mapping")

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "height_cm", height_cm)
        object.__setattr__(
            self,
            "confidence",
            _bounded_score(self.confidence, "height estimate confidence"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "height_cm": round(self.height_cm, 2),
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
            "metadata": dict(self.metadata),
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