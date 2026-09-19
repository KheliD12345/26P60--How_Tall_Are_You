"""Interfaces and lightweight adapters for human body detection.

Heavy pose and head models are deliberately supplied by dependency injection.
This module only defines their target-side contract and normalises their output.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .advanced_models import BodyDetections, Landmark
from .models import PersonEndpoints
from .person import detect_person_endpoints


class DetectionStatus(str, Enum):
    SUCCESS = "success"
    NO_PERSON = "no_person"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    PARTIAL = "partial"


class ImageDetector(Protocol):
    def detect(self, image: np.ndarray) -> object:
        """Return detector-specific output for one BGR image."""


_COORDINATE_SYSTEMS = {
    "normalized": "normalized",
    "normalised": "normalized",
    "pixel": "pixel",
    "pixels": "pixel",
}

_POSE_METADATA_KEYS = {
    "coordinate_system",
    "normalized",
    "normalised",
    "success",
    "status",
    "message",
    "error",
    "metadata",
    "total_keypoints",
}


def _coordinate_system(value: object | None) -> str:
    if value is None:
        return "normalized"
    try:
        return _COORDINATE_SYSTEMS[str(value).lower()]
    except KeyError as error:
        raise ValueError(
            "coordinate system must be normalized or pixel"
        ) from error


def _normalise_landmark(
    value: object,
    coordinate_system: str = "normalized",
) -> Landmark:
    system = _coordinate_system(coordinate_system)
    if isinstance(value, Mapping):
        local_system = value.get("coordinate_system")
        if local_system is None and "normalized" in value:
            local_system = "normalized" if value["normalized"] else "pixel"
        if local_system is not None:
            system = _coordinate_system(local_system)
    else:
        local_system = getattr(value, "coordinate_system", None)
        if local_system is None and hasattr(value, "normalized"):
            local_system = "normalized" if value.normalized else "pixel"
        if local_system is not None:
            system = _coordinate_system(local_system)
    return Landmark.from_value(
        value,
        normalized=system == "normalized",
        coordinate_system=system,
    )


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _landmark_mapping(
    raw: Mapping[str, Any],
    coordinate_system: str,
) -> dict[str, Landmark]:
    values = raw.get("keypoints", raw.get("landmarks", raw))
    if not isinstance(values, Mapping):
        raise ValueError("pose keypoints must be a mapping")

    keypoints: dict[str, Landmark] = {}
    for name, value in values.items():
        if str(name) in _POSE_METADATA_KEYS:
            continue
        if str(name) == "heel_landmarks":
            if not isinstance(value, Mapping):
                raise ValueError("heel_landmarks must be a mapping")
            for side, heel in value.items():
                if str(side) in {"left", "right"} and heel is not None:
                    keypoints[f"{side}_heel"] = _normalise_landmark(
                        heel,
                        coordinate_system,
                    )
            continue
        keypoints[str(name)] = _normalise_landmark(value, coordinate_system)
    return keypoints


def normalise_pose_keypoints(
    value: object,
    *,
    coordinate_system: str | None = None,
) -> BodyDetections:
    """Convert named pose output into the stable body-detection contract."""
    if isinstance(value, BodyDetections):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("pose output must be a mapping")

    source_system = _coordinate_system(
        coordinate_system
        or value.get("coordinate_system")
        or ("normalized" if value.get("normalized", True) else "pixel")
    )
    keypoints = _landmark_mapping(value, source_system)
    return BodyDetections(
        keypoints=keypoints,
        head_top=keypoints.get("head_top"),
        head_bottom=keypoints.get("head_bottom"),
    )


def _normalise_bbox(
    value: object,
    coordinate_system: str,
) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        try:
            bbox = tuple(
                float(value[name]) for name in ("x1", "y1", "x2", "y2")
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "head bounding box requires x1, y1, x2, and y2"
            ) from error
    else:
        try:
            bbox = tuple(float(item) for item in value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError(
                "head bounding box must contain four numeric values"
            ) from error
        if len(bbox) != 4:
            raise ValueError("head bounding box must contain four values")

    if not all(isfinite(item) for item in bbox):
        raise ValueError("head bounding box values must be finite")
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise ValueError("head bounding box must have non-negative dimensions")
    _coordinate_system(coordinate_system)
    return bbox


def normalise_head_detection(
    value: object,
    *,
    coordinate_system: str | None = None,
) -> BodyDetections:
    """Convert VGGHeads-like output into ``BodyDetections``."""
    if isinstance(value, BodyDetections):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("head output must be a mapping")

    detection = value.get("head_detection", value)
    if not isinstance(detection, Mapping):
        raise ValueError("head detection must be a mapping")

    bbox_value = detection.get("head_bbox_pixels")
    bbox_system = "pixel" if bbox_value is not None else (
        coordinate_system
        or detection.get("head_bbox_coordinate_system")
        or detection.get("coordinate_system")
        or "normalized"
    )
    if bbox_value is None:
        bbox_value = detection.get("head_bbox", detection.get("bbox"))
    if bbox_value is None:
        return BodyDetections()

    bbox = _normalise_bbox(bbox_value, bbox_system)
    system = _coordinate_system(bbox_system)
    center_x = (bbox[0] + bbox[2]) / 2.0
    top_value = _mapping_value(
        detection,
        "head_top",
        "head_top_y_pixels" if system == "pixel" else "head_top_y",
    )
    bottom_value = _mapping_value(
        detection,
        "head_bottom",
        "head_bottom_y_pixels" if system == "pixel" else "head_bottom_y",
    )

    if isinstance(top_value, (Mapping, tuple, list)):
        head_top = _normalise_landmark(top_value, system)
    else:
        head_top = Landmark(
            center_x,
            bbox[1] if top_value is None else float(top_value),
            coordinate_system=system,
        )
    if isinstance(bottom_value, (Mapping, tuple, list)):
        head_bottom = _normalise_landmark(bottom_value, system)
    else:
        head_bottom = Landmark(
            center_x,
            bbox[3] if bottom_value is None else float(bottom_value),
            coordinate_system=system,
        )

    confidence = _mapping_value(
        detection,
        "head_confidence",
        "confidence",
        "score",
    )
    return BodyDetections(
        head_top=head_top,
        head_bottom=head_bottom,
        head_bbox=bbox,
        head_confidence=0.0 if confidence is None else float(confidence),
        head_bbox_coordinate_system=system,
    )


def body_detections_from_person(person: PersonEndpoints) -> BodyDetections:
    """Represent HOG endpoints without inventing intermediate pose points."""
    confidence = max(0.0, min(1.0, float(person.score)))
    bottom = Landmark(
        person.bottom_of_feet[0],
        person.bottom_of_feet[1],
        visibility=confidence,
        coordinate_system="pixel",
    )
    return BodyDetections(
        keypoints={"left_heel": bottom, "right_heel": bottom},
        head_top=Landmark(
            person.top_of_head[0],
            person.top_of_head[1],
            visibility=confidence,
            coordinate_system="pixel",
        ),
    )


def merge_body_detections(
    primary: BodyDetections,
    secondary: BodyDetections,
) -> BodyDetections:
    """Fill missing fields from ``secondary`` while retaining valid values."""
    keypoints = dict(secondary.keypoints)
    for name, landmark in primary.keypoints.items():
        other = keypoints.get(name)
        if other is not None and other.coordinate_system != landmark.coordinate_system:
            raise ValueError(f"coordinate system mismatch for keypoint: {name}")
        keypoints[name] = landmark

    if (
        primary.head_bbox is not None
        and secondary.head_bbox is not None
        and primary.head_bbox_coordinate_system
        != secondary.head_bbox_coordinate_system
    ):
        raise ValueError("head bounding box coordinate systems do not match")

    return BodyDetections(
        keypoints=keypoints,
        head_top=primary.head_top or secondary.head_top,
        head_bottom=primary.head_bottom or secondary.head_bottom,
        head_bbox=primary.head_bbox or secondary.head_bbox,
        head_confidence=(
            primary.head_confidence
            if primary.head_bbox is not None
            else secondary.head_confidence
        ),
        hair_top=primary.hair_top or secondary.hair_top,
        hair_bottom=primary.hair_bottom or secondary.hair_bottom,
        hand_lengths=primary.hand_lengths or secondary.hand_lengths,
        segmentation_mask=(
            primary.segmentation_mask
            if primary.segmentation_mask is not None
            else secondary.segmentation_mask
        ),
        head_bbox_coordinate_system=(
            primary.head_bbox_coordinate_system
            if primary.head_bbox is not None
            else secondary.head_bbox_coordinate_system
        ),
    )


@dataclass(frozen=True)
class DetectorResult:
    detector: str
    status: DetectionStatus | str
    detections: BodyDetections = field(default_factory=BodyDetections)
    diagnostics: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", DetectionStatus(self.status))
        if not isinstance(self.detections, BodyDetections):
            raise TypeError("detector detections must be BodyDetections")
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, str) or not item.strip() for item in diagnostics):
            raise ValueError("detector diagnostics must be non-empty strings")
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def available(self) -> bool:
        return self.status in {DetectionStatus.SUCCESS, DetectionStatus.PARTIAL}

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "status": self.status.value,
            "diagnostics": list(self.diagnostics),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BodyDetectionResult:
    detections: BodyDetections = field(default_factory=BodyDetections)
    detector_results: Mapping[str, DetectorResult] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.detections, BodyDetections):
            raise TypeError("body detections must be BodyDetections")
        results = dict(self.detector_results)
        if any(not isinstance(result, DetectorResult) for result in results.values()):
            raise TypeError("detector results must contain DetectorResult values")
        object.__setattr__(self, "detector_results", results)
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, str) or not item.strip() for item in diagnostics):
            raise ValueError("body diagnostics must be non-empty strings")
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def status(self) -> DetectionStatus:
        statuses = [result.status for result in self.detector_results.values()]
        if any(result.available for result in self.detector_results.values()):
            return (
                DetectionStatus.SUCCESS
                if statuses and all(status == DetectionStatus.SUCCESS for status in statuses)
                else DetectionStatus.PARTIAL
            )
        if statuses and all(status == DetectionStatus.NO_PERSON for status in statuses):
            return DetectionStatus.NO_PERSON
        if statuses and all(status == DetectionStatus.UNAVAILABLE for status in statuses):
            return DetectionStatus.UNAVAILABLE
        if any(status == DetectionStatus.FAILED for status in statuses):
            return DetectionStatus.FAILED
        return DetectionStatus.UNAVAILABLE

    @property
    def success(self) -> bool:
        return self.status in {DetectionStatus.SUCCESS, DetectionStatus.PARTIAL}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "detectors": {
                name: result.to_dict()
                for name, result in self.detector_results.items()
            },
            "diagnostics": list(self.diagnostics),
            "keypoints": {
                name: {
                    "x": landmark.x,
                    "y": landmark.y,
                    "visibility": landmark.visibility,
                    "coordinate_system": landmark.coordinate_system,
                }
                for name, landmark in self.detections.keypoints.items()
            },
        }


class _NormalisingAdapter:
    def __init__(
        self,
        detector: Callable[[np.ndarray], object] | ImageDetector,
        name: str,
    ):
        self._detector = detector
        self.name = name

    def _run(self, image: np.ndarray) -> object:
        detect = getattr(self._detector, "detect", self._detector)
        return detect(image)

    def _result(
        self,
        image: np.ndarray,
        normaliser: Callable[[object], BodyDetections],
    ) -> DetectorResult:
        try:
            raw = self._run(image)
        except ImportError as error:
            return DetectorResult(
                self.name,
                DetectionStatus.UNAVAILABLE,
                diagnostics=(f"{self.name} is unavailable: {error}",),
            )
        except Exception as error:
            return DetectorResult(
                self.name,
                DetectionStatus.FAILED,
                diagnostics=(f"{self.name} failed: {error}",),
            )

        if isinstance(raw, DetectorResult):
            return raw
        if isinstance(raw, Mapping) and raw.get("status") in {
            DetectionStatus.UNAVAILABLE.value,
            DetectionStatus.FAILED.value,
        }:
            status = DetectionStatus(raw["status"])
            message = str(raw.get("error", raw.get("message", status.value)))
            return DetectorResult(self.name, status, diagnostics=(message,))
        if isinstance(raw, Mapping) and raw.get("success") is False:
            return DetectorResult(
                self.name,
                DetectionStatus.FAILED,
                diagnostics=(str(raw.get("error", f"{self.name} failed")),),
            )

        try:
            detections = normaliser(raw)
        except (TypeError, ValueError) as error:
            return DetectorResult(
                self.name,
                DetectionStatus.FAILED,
                diagnostics=(f"invalid {self.name} output: {error}",),
            )
        status = (
            DetectionStatus.SUCCESS
            if self._has_detection(detections)
            else DetectionStatus.NO_PERSON
        )
        metadata: Mapping[str, Any] = {}
        if isinstance(raw, Mapping) and isinstance(raw.get("metadata"), Mapping):
            metadata = raw["metadata"]
        metadata = dict(metadata)
        metadata.setdefault("total_keypoints", len(detections.keypoints))
        return DetectorResult(
            self.name,
            status,
            detections=detections,
            metadata=metadata,
        )

    @staticmethod
    def _has_detection(detections: BodyDetections) -> bool:
        return bool(
            detections.keypoints
            or detections.head_top
            or detections.head_bottom
            or detections.head_bbox is not None
        )


class PoseDetectorAdapter(_NormalisingAdapter):
    def __init__(self, detector: Callable[[np.ndarray], object] | ImageDetector):
        super().__init__(detector, "pose")

    def detect(self, image: np.ndarray) -> DetectorResult:
        return self._result(image, normalise_pose_keypoints)


class HeadDetectorAdapter(_NormalisingAdapter):
    def __init__(self, detector: Callable[[np.ndarray], object] | ImageDetector):
        super().__init__(detector, "head")

    def detect(self, image: np.ndarray) -> DetectorResult:
        return self._result(image, normalise_head_detection)


class PersonFallbackAdapter:
    """Use the existing HOG/GrabCut detector as a pose-independent fallback."""

    name = "person_fallback"

    def detect(self, image: np.ndarray) -> DetectorResult:
        try:
            person = detect_person_endpoints(image)
        except ImportError as error:
            return DetectorResult(
                self.name,
                DetectionStatus.UNAVAILABLE,
                diagnostics=(f"{self.name} is unavailable: {error}",),
            )
        except Exception as error:
            return DetectorResult(
                self.name,
                DetectionStatus.FAILED,
                diagnostics=(f"{self.name} failed: {error}",),
            )
        if person is None:
            return DetectorResult(self.name, DetectionStatus.NO_PERSON)
        return DetectorResult(
            self.name,
            DetectionStatus.SUCCESS,
            detections=body_detections_from_person(person),
            metadata={"source": "hog_grabcut"},
        )


class UnavailableDetector:
    """Explicit result for an optional detector that is not configured."""

    def __init__(self, name: str, reason: str = "optional detector is not configured"):
        self.name = name
        self.reason = reason

    def detect(self, image: np.ndarray) -> DetectorResult:
        del image
        return DetectorResult(
            self.name,
            DetectionStatus.UNAVAILABLE,
            diagnostics=(f"{self.name} unavailable: {self.reason}",),
        )


def build_body_detection_result(
    results: Mapping[str, DetectorResult],
) -> BodyDetectionResult:
    """Merge detector results without allowing empty outputs to erase data."""
    detections = BodyDetections()
    diagnostics: list[str] = []
    for name, result in results.items():
        if not isinstance(result, DetectorResult):
            raise TypeError(f"{name} result must be a DetectorResult")
        diagnostics.extend(result.diagnostics)
        if result.available:
            detections = merge_body_detections(detections, result.detections)
    return BodyDetectionResult(
        detections=detections,
        detector_results=results,
        diagnostics=tuple(diagnostics),
    )


# American spelling is kept as a small compatibility alias for callers.
normalize_pose_keypoints = normalise_pose_keypoints
normalize_head_detection = normalise_head_detection
BodyPoseDetector = PoseDetectorAdapter
HeadDetector = HeadDetectorAdapter