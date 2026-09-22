"""Interfaces and lightweight adapters for human body detection.

Heavy pose and head models are deliberately supplied by dependency injection.
This module only defines their target-side contract and normalises their output.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import cv2
import numpy as np

from .advanced_models import BodyDetections, Landmark
from .models import PersonEndpoints
from .optional_detectors import (
    LazyDetector,
    OptionalDetectorUnavailable,
    import_optional_module,
)
from .person import detect_person_endpoints


class DetectionStatus(str, Enum):
    SUCCESS = "success"
    NO_PERSON = "no_person"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass(frozen=True)
class BodyDetectionConfig:
    """Explicit optional detector configuration for body detection."""

    pose_detector: object | None = None
    head_detector: object | None = None
    segmentation_detector: object | None = None
    hand_detector: object | None = None


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


def _mask_array(value: object) -> np.ndarray:
    try:
        mask = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("segmentation mask must be array-like") from error
    if mask.ndim < 2:
        raise ValueError("segmentation mask must have at least two dimensions")
    if mask.dtype.kind not in "biuf":
        raise ValueError("segmentation mask must contain numeric values")
    if mask.dtype.kind == "f" and not np.isfinite(mask).all():
        raise ValueError("segmentation mask values must be finite")
    return mask.copy()


def _segmentation_payload(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("segmentation output must be a mapping")
    payload = value.get("segmentation", value)
    if not isinstance(payload, Mapping):
        raise ValueError("segmentation data must be a mapping")
    return payload


def _endpoint_from_value(
    value: object,
    *,
    mask: np.ndarray | None,
    coordinate_system: str,
    top: bool,
) -> Landmark:
    system = _coordinate_system(coordinate_system)
    local_value = value
    local_system = system
    if isinstance(value, Mapping):
        local_system = value.get("coordinate_system", system)
        if "coordinate_system" not in value and "normalized" in value:
            local_system = "normalized" if value["normalized"] else "pixel"
        local_system = _coordinate_system(local_system)
        y_value = value.get("y")
        if y_value is None:
            raise ValueError("hair endpoint requires a y coordinate")
        x_value = value.get("x")
        if x_value is None:
            local_value = y_value
        else:
            return _normalise_landmark(value, local_system)
    elif isinstance(value, (tuple, list)):
        return _normalise_landmark(value, local_system)

    try:
        y_value = float(local_value)
    except (TypeError, ValueError) as error:
        raise ValueError("hair endpoint y coordinate must be numeric") from error
    x_value = 0.0
    if mask is not None:
        row = y_value
        if local_system == "normalized":
            row *= max(0, mask.shape[0] - 1)
        target_row = max(0, min(mask.shape[0] - 1, int(round(row))))
        row_order = sorted(
            range(mask.shape[0]),
            key=lambda candidate: abs(candidate - target_row),
        )
        columns = np.array([], dtype=int)
        for candidate in row_order:
            row_values = mask[candidate]
            if row_values.ndim > 1:
                row_values = np.any(
                    row_values != 0,
                    axis=tuple(range(1, row_values.ndim)),
                )
            columns = np.flatnonzero(row_values != 0)
            if columns.size:
                break
        if columns.size:
            x_value = float(columns.mean())
            if local_system == "normalized":
                x_value /= max(1, mask.shape[1] - 1)
    return Landmark(x_value, y_value, coordinate_system=local_system)


def normalise_segmentation(
    value: object,
    *,
    mask_coordinate_system: str | None = None,
    landmark_coordinate_system: str | None = None,
) -> BodyDetections:
    """Convert body or hair segmentation output into ``BodyDetections``."""
    if isinstance(value, BodyDetections):
        return value
    payload = _segmentation_payload(value)
    raw_mask = payload.get("segmentation_mask", payload.get("mask"))
    mask = None if raw_mask is None else _mask_array(raw_mask)

    hair = payload.get("hair_length", payload.get("hair", {}))
    if hair is None:
        hair = {}
    if not isinstance(hair, Mapping):
        raise ValueError("hair endpoints must be a mapping")
    top_value = payload.get("hair_top", hair.get("top"))
    bottom_value = payload.get("hair_bottom", hair.get("bottom"))
    landmark_system = _coordinate_system(
        landmark_coordinate_system
        or payload.get("landmark_coordinate_system")
        or payload.get("hair_coordinate_system")
        or "normalized"
    )
    return BodyDetections(
        segmentation_mask=mask,
        hair_top=(
            None
            if top_value is None
            else _endpoint_from_value(
                top_value,
                mask=mask,
                coordinate_system=landmark_system,
                top=True,
            )
        ),
        hair_bottom=(
            None
            if bottom_value is None
            else _endpoint_from_value(
                bottom_value,
                mask=mask,
                coordinate_system=landmark_system,
                top=False,
            )
        ),
    )


def _hand_payload(value: object) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("hand output must be a mapping")
    payload = value.get("hands", value.get("hand_detections", value))
    if payload == {}:
        return value, {"hands": []}
    if not isinstance(payload, (list, tuple)):
        raise ValueError("hands must be a list")
    return value, {"hands": payload}


def _hand_label(value: object) -> str:
    if value is None:
        return "unknown"
    label = str(value).strip().lower()
    if label in {"left", "right"}:
        return label
    if label in {"unknown", "none", ""}:
        return "unknown"
    raise ValueError("handedness must be left, right, or unknown")


def _hand_pair(
    hand: Mapping[str, Any],
    coordinate_system: str,
) -> tuple[Landmark, Landmark]:
    length = hand.get("hand_length", hand.get("hand_lengths"))
    if isinstance(length, Mapping):
        first = length.get("landmark_0", length.get("wrist"))
        second = length.get(
            "landmark_12",
            length.get("middle_tip", length.get("fingertip")),
        )
    elif isinstance(length, (list, tuple)) and len(length) == 2:
        first, second = length
    else:
        all_landmarks = hand.get("all_landmarks", hand.get("landmarks", {}))
        if not isinstance(all_landmarks, Mapping):
            raise ValueError("hand length endpoints are missing")
        first = all_landmarks.get("landmark_0", all_landmarks.get("wrist"))
        second = all_landmarks.get(
            "landmark_12",
            all_landmarks.get("middle_tip", all_landmarks.get("fingertip")),
        )
    if first is None or second is None:
        raise ValueError("hand length requires wrist and middle fingertip")
    return (
        _normalise_landmark(first, coordinate_system),
        _normalise_landmark(second, coordinate_system),
    )


def normalise_hand_detection(
    value: object,
    *,
    coordinate_system: str | None = None,
) -> BodyDetections:
    """Convert MediaPipe-style hand output without silently flipping labels."""
    if isinstance(value, BodyDetections):
        return value
    root, payload = _hand_payload(value)
    convention = str(
        root.get("handedness_convention", "subject")
    ).lower()
    if convention in {"image", "image_perspective", "mediapipe"}:
        raise ValueError(
            "image-perspective handedness must be corrected before normalisation"
        )
    if root.get("labels_are_subject_relative") is False:
        raise ValueError("handedness labels must be subject-relative")
    system = _coordinate_system(
        coordinate_system
        or root.get("coordinate_system")
        or ("normalized" if root.get("normalized", True) else "pixel")
    )
    hands = payload["hands"]
    pairs: list[tuple[Landmark, Landmark]] = []
    labels: list[str] = []
    confidences: list[float] = []
    for hand in hands:
        if not isinstance(hand, Mapping):
            raise ValueError("each hand detection must be a mapping")
        pairs.append(_hand_pair(hand, system))
        labels.append(_hand_label(hand.get("handedness", hand.get("label"))))
        confidence = hand.get("confidence", hand.get("score", 1.0))
        try:
            confidences.append(float(confidence))
        except (TypeError, ValueError) as error:
            raise ValueError("hand confidence must be numeric") from error
    return BodyDetections(
        hand_lengths=tuple(pairs),
        handedness=tuple(labels),
        hand_confidences=tuple(confidences),
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
    hand_lengths = primary.hand_lengths or secondary.hand_lengths
    hand_source = primary if primary.hand_lengths else secondary
    for name, landmark in primary.keypoints.items():
        other = keypoints.get(name)
        if other is not None and other.coordinate_system != landmark.coordinate_system:
            raise ValueError(f"coordinate system mismatch for keypoint: {name}")
        keypoints[name] = landmark

    head_top = primary.head_top or secondary.head_top
    head_bottom = primary.head_bottom or secondary.head_bottom
    head_bbox = primary.head_bbox or secondary.head_bbox
    head_bbox_coordinate_system = (
        primary.head_bbox_coordinate_system
        if primary.head_bbox is not None
        else secondary.head_bbox_coordinate_system
    )
    hair_top = primary.hair_top or secondary.hair_top
    hair_bottom = primary.hair_bottom or secondary.hair_bottom
    _validate_coordinate_systems("keypoints", tuple(keypoints.values()))
    _validate_coordinate_systems("head endpoints", (head_top, head_bottom))
    if head_bbox is not None:
        _validate_coordinate_systems(
            "head detection",
            (head_top, head_bottom),
            head_bbox_coordinate_system,
        )
    _validate_coordinate_systems("hair endpoints", (hair_top, hair_bottom))
    _validate_coordinate_systems(
        "hand landmarks",
        tuple(point for pair in hand_lengths for point in pair),
    )

    return BodyDetections(
        keypoints=keypoints,
        head_top=head_top,
        head_bottom=head_bottom,
        head_bbox=head_bbox,
        head_confidence=(
            primary.head_confidence
            if primary.head_bbox is not None
            else secondary.head_confidence
        ),
        hair_top=hair_top,
        hair_bottom=hair_bottom,
        hand_lengths=hand_lengths,
        handedness=hand_source.handedness,
        hand_confidences=hand_source.hand_confidences,
        segmentation_mask=(
            primary.segmentation_mask
            if primary.segmentation_mask is not None
            else secondary.segmentation_mask
        ),
        head_bbox_coordinate_system=head_bbox_coordinate_system,
    )


def _validate_coordinate_systems(
    field_name: str,
    landmarks: tuple[Landmark | None, ...],
    extra_system: str | None = None,
) -> None:
    systems = {
        landmark.coordinate_system
        for landmark in landmarks
        if landmark is not None
    }
    if extra_system is not None:
        systems.add(extra_system)
    if len(systems) > 1:
        raise ValueError(f"coordinate system mismatch for {field_name}")


def _landmark_to_dict(landmark: Landmark | None) -> dict[str, Any] | None:
    if landmark is None:
        return None
    return {
        "x": landmark.x,
        "y": landmark.y,
        "visibility": landmark.visibility,
        "coordinate_system": landmark.coordinate_system,
    }


def _serialise_detection_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _serialise_detection_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_serialise_detection_value(item) for item in value]
    return value


def _detections_to_dict(detections: BodyDetections) -> dict[str, Any]:
    return {
        "keypoints": {
            name: _landmark_to_dict(landmark)
            for name, landmark in detections.keypoints.items()
        },
        "head_top": _landmark_to_dict(detections.head_top),
        "head_bottom": _landmark_to_dict(detections.head_bottom),
        "head_bbox": (
            None
            if detections.head_bbox is None
            else list(detections.head_bbox)
        ),
        "head_bbox_coordinate_system": detections.head_bbox_coordinate_system,
        "head_confidence": detections.head_confidence,
        "hair_top": _landmark_to_dict(detections.hair_top),
        "hair_bottom": _landmark_to_dict(detections.hair_bottom),
        "hand_lengths": [
            [_landmark_to_dict(first), _landmark_to_dict(second)]
            for first, second in detections.hand_lengths
        ],
        "handedness": list(detections.handedness),
        "hand_confidences": list(detections.hand_confidences),
        "segmentation_mask": _serialise_detection_value(
            detections.segmentation_mask
        ),
    }


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
            "detections": _detections_to_dict(self.detections),
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
            "detections": _detections_to_dict(self.detections),
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
        except OptionalDetectorUnavailable as error:
            return DetectorResult(
                self.name,
                DetectionStatus.UNAVAILABLE,
                diagnostics=(f"{self.name} is unavailable: {error}",),
            )
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
        if detections.segmentation_mask is not None:
            metadata.setdefault("mask_shape", tuple(detections.segmentation_mask.shape))
            if isinstance(raw, Mapping):
                metadata.setdefault(
                    "mask_coordinate_system",
                    raw.get("mask_coordinate_system", "pixel"),
                )
        if detections.hand_lengths:
            metadata.setdefault("hand_count", len(detections.hand_lengths))
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
            or detections.segmentation_mask is not None
            or detections.hair_top
            or detections.hair_bottom
            or bool(detections.hand_lengths)
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


_VITPOSE_INDEX_NAMES = {
    "landmark_11": "left_shoulder",
    "landmark_12": "right_shoulder",
    "landmark_23": "left_hip",
    "landmark_24": "right_hip",
    "landmark_25": "left_knee",
    "landmark_26": "right_knee",
    "landmark_27": "left_ankle",
    "landmark_28": "right_ankle",
    "landmark_29": "left_heel",
    "landmark_30": "right_heel",
}

_VITPOSE_GROUPS = {
    "shoulder_width": {
        "landmark_11": "left_shoulder",
        "landmark_12": "right_shoulder",
    },
    "hip_width": {
        "landmark_23": "left_hip",
        "landmark_24": "right_hip",
    },
    "upper_leg_length": {
        "landmark_23": "left_hip",
        "landmark_25": "left_knee",
        "landmark_24": "right_hip",
        "landmark_26": "right_knee",
    },
    "lower_leg_length": {
        "landmark_25": "left_knee",
        "landmark_27": "left_ankle",
        "landmark_26": "right_knee",
        "landmark_28": "right_ankle",
    },
    "upper_arm_length": {
        "landmark_11": "left_shoulder",
        "landmark_13": "left_elbow",
        "landmark_12": "right_shoulder",
        "landmark_14": "right_elbow",
    },
    "forearm_length": {
        "landmark_13": "left_elbow",
        "landmark_15": "left_wrist",
        "landmark_14": "right_elbow",
        "landmark_16": "right_wrist",
    },
    "shoulder_to_waist": {
        "landmark_11": "left_shoulder",
        "landmark_23": "left_hip",
        "landmark_12": "right_shoulder",
        "landmark_24": "right_hip",
    },
}

_VITPOSE_DIRECT_NAMES = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "nose",
    "head",
}


def _is_landmark_record(value: object) -> bool:
    if isinstance(value, Mapping):
        return "x" in value and "y" in value
    if isinstance(value, (tuple, list)):
        return len(value) >= 2
    return hasattr(value, "x") and hasattr(value, "y")


def _add_vitpose_landmark(
    keypoints: dict[str, object],
    name: str,
    value: object,
) -> None:
    if value is None:
        return
    if not _is_landmark_record(value):
        raise ValueError(f"ViTPose landmark {name} is malformed")
    keypoints[name] = value


def _flatten_vitpose_output(value: object) -> object:
    if isinstance(value, BodyDetections):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("ViTPose output must be a mapping")

    source = value.get("keypoints", value.get("landmarks", value))
    if not isinstance(source, Mapping):
        raise ValueError("ViTPose keypoints must be a mapping")

    keypoints: dict[str, object] = {}
    for raw_name, raw_value in source.items():
        name = str(raw_name)
        if name in _POSE_METADATA_KEYS or name in {"coordinate_system", "metadata"}:
            continue
        if name == "heel_landmarks":
            if raw_value is None:
                continue
            if not isinstance(raw_value, Mapping):
                raise ValueError("ViTPose heel_landmarks must be a mapping")
            _add_vitpose_landmark(
                keypoints,
                "left_heel",
                raw_value.get("left"),
            )
            _add_vitpose_landmark(
                keypoints,
                "right_heel",
                raw_value.get("right"),
            )
            continue
        if name in _VITPOSE_GROUPS:
            if raw_value is None:
                continue
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"ViTPose group {name} must be a mapping")
            group_names = _VITPOSE_GROUPS[name]
            for source_name, target_name in group_names.items():
                if source_name in raw_value:
                    _add_vitpose_landmark(
                        keypoints,
                        target_name,
                        raw_value[source_name],
                    )
            for side in ("left", "right"):
                side_values = raw_value.get(side)
                if side_values is None:
                    continue
                if not isinstance(side_values, Mapping):
                    raise ValueError(
                        f"ViTPose group {name}.{side} must be a mapping"
                    )
                for source_name, target_name in group_names.items():
                    if source_name in side_values:
                        _add_vitpose_landmark(
                            keypoints,
                            target_name,
                            side_values[source_name],
                        )
            continue
        target_name = _VITPOSE_INDEX_NAMES.get(name, name)
        if name in _VITPOSE_INDEX_NAMES or name in _VITPOSE_DIRECT_NAMES:
            _add_vitpose_landmark(keypoints, target_name, raw_value)

    normalised: dict[str, Any] = {
        "coordinate_system": "normalized",
        "keypoints": keypoints,
    }
    if isinstance(value.get("metadata"), Mapping):
        normalised["metadata"] = dict(value["metadata"])
    return normalised


def _optional_class_factory(
    module_name: str,
    class_name: str,
    options: Mapping[str, Any],
) -> Callable[[], object]:
    def factory() -> object:
        module = import_optional_module(module_name)
        try:
            detector_class = getattr(module, class_name)
        except AttributeError as error:
            raise OptionalDetectorUnavailable(
                f"optional module {module_name!r} has no {class_name} detector"
            ) from error
        if not callable(detector_class):
            raise OptionalDetectorUnavailable(
                f"{class_name} is not a callable detector"
            )
        return detector_class(**dict(options))

    return factory


class ViTPoseDetectorAdapter(PoseDetectorAdapter):
    """Lazily load and normalise a ViTPose-style grouped detector."""

    def __init__(
        self,
        detector_factory: Callable[[], object] | None = None,
        *,
        model_name: str = "l",
        yolo_model: str = "yolov8s",
        device: str | None = None,
        yolo_size: int = 320,
        detector_options: Mapping[str, Any] | None = None,
        module_name: str = "vitpose_detection.pose_detection",
        detector_class_name: str = "PoseDetector",
    ) -> None:
        options: dict[str, Any] = {
            "model_name": model_name,
            "yolo_model": yolo_model,
            "device": device,
            "yolo_size": yolo_size,
        }
        if detector_options is not None:
            options.update(detector_options)
        factory = detector_factory or _optional_class_factory(
            module_name,
            detector_class_name,
            options,
        )
        self.lazy_detector = LazyDetector(factory, name="pose")
        super().__init__(self.lazy_detector)

    def _run(self, image: np.ndarray) -> object:
        raw = self.lazy_detector.detect(image)
        if isinstance(raw, Mapping) and (
            raw.get("status") in {
                DetectionStatus.UNAVAILABLE.value,
                DetectionStatus.FAILED.value,
            }
            or raw.get("success") is False
        ):
            return raw
        return _flatten_vitpose_output(raw)


class VGGHeadsDetectorAdapter(HeadDetectorAdapter):
    """Lazily load and normalise a VGGHeads-style head detector."""

    def __init__(
        self,
        detector_factory: Callable[[], object] | None = None,
        *,
        model: str = "vgg_heads_l",
        conf_threshold: float = 0.5,
        detector_options: Mapping[str, Any] | None = None,
        module_name: str = "vggheads_detection.head_detection",
        detector_class_name: str = "HeadDetector",
    ) -> None:
        options: dict[str, Any] = {
            "model": model,
            "conf_threshold": conf_threshold,
        }
        if detector_options is not None:
            options.update(detector_options)
        factory = detector_factory or _optional_class_factory(
            module_name,
            detector_class_name,
            options,
        )
        self.lazy_detector = LazyDetector(factory, name="head")
        super().__init__(self.lazy_detector)

    def _run(self, image: np.ndarray) -> object:
        raw = self.lazy_detector.detect(image)
        if isinstance(raw, Mapping):
            if raw.get("status") in {
                DetectionStatus.UNAVAILABLE.value,
                DetectionStatus.FAILED.value,
            } or raw.get("success") is False:
                return raw
            has_bbox = raw.get("head_bbox") is not None or raw.get(
                "head_bbox_pixels"
            ) is not None
            if raw.get("head_detected") is True and not has_bbox:
                raise ValueError(
                    "VGGHeads output marks a head as detected without a bounding box"
                )
        return raw


LazyPoseDetectorAdapter = ViTPoseDetectorAdapter
LazyHeadDetectorAdapter = VGGHeadsDetectorAdapter

class SegmentationDetectorAdapter(_NormalisingAdapter):
    def __init__(self, detector: Callable[[np.ndarray], object] | ImageDetector):
        super().__init__(detector, "segmentation")

    def detect(self, image: np.ndarray) -> DetectorResult:
        return self._result(image, normalise_segmentation)


class HandDetectorAdapter(_NormalisingAdapter):
    def __init__(self, detector: Callable[[np.ndarray], object] | ImageDetector):
        super().__init__(detector, "hands")

    def detect(self, image: np.ndarray) -> DetectorResult:
        return self._result(image, normalise_hand_detection)


class MediaPipeSegmentationAdapter(SegmentationDetectorAdapter):
    """Lazily load a MediaPipe hair segmenter and always request its mask."""

    def __init__(
        self,
        detector_factory: Callable[[], object] | None = None,
        *,
        model_path: str | Path | None = None,
        use_face_detection: bool = True,
        detector_options: Mapping[str, Any] | None = None,
        module_name: str = "mediapipe_detection.hair_segmentation",
        detector_class_name: str = "HairSegmenter",
    ) -> None:
        options: dict[str, Any] = {
            "model_path": model_path,
            "use_face_detection": use_face_detection,
        }
        if detector_options is not None:
            options.update(detector_options)
        factory = detector_factory or _optional_class_factory(
            module_name,
            detector_class_name,
            options,
        )
        self.lazy_detector = LazyDetector(factory, name="segmentation")
        super().__init__(self.lazy_detector)

    def _run(self, image: np.ndarray) -> object:
        raw = self.lazy_detector.invoke("segment", image, return_mask=True)
        if isinstance(raw, Mapping) and (
            raw.get("status") in {
                DetectionStatus.UNAVAILABLE.value,
                DetectionStatus.FAILED.value,
            }
            or raw.get("success") is False
        ):
            return raw
        if not isinstance(raw, Mapping):
            return raw

        payload = dict(raw)
        nested = payload.get("segmentation")
        if isinstance(nested, Mapping):
            nested_payload = dict(nested)
            if "mask" in nested_payload or "segmentation_mask" in nested_payload:
                nested_payload.setdefault("mask_coordinate_system", "pixel")
                payload.setdefault("mask_coordinate_system", "pixel")
            payload["segmentation"] = nested_payload
        elif "mask" in payload or "segmentation_mask" in payload:
            payload.setdefault("mask_coordinate_system", "pixel")
        metadata = payload.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        mask_source = nested if isinstance(nested, Mapping) else payload
        raw_mask = mask_source.get(
            "mask",
            mask_source.get("segmentation_mask"),
        )
        if raw_mask is not None:
            metadata.setdefault("mask_coordinate_system", "pixel")
            try:
                metadata.setdefault("mask_shape", tuple(np.asarray(raw_mask).shape))
            except (TypeError, ValueError):
                pass
        payload["metadata"] = metadata
        return payload


class MediaPipeHandDetectorAdapter(HandDetectorAdapter):
    """Lazily load a MediaPipe hand detector with subject-relative labels."""

    def __init__(
        self,
        detector_factory: Callable[[], object] | None = None,
        *,
        model_path: str | Path | None = None,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        detector_options: Mapping[str, Any] | None = None,
        module_name: str = "mediapipe_detection.hand_detection",
        detector_class_name: str = "HandLandmarkDetector",
    ) -> None:
        options: dict[str, Any] = {
            "model_path": model_path,
            "num_hands": num_hands,
            "min_hand_detection_confidence": min_hand_detection_confidence,
            "min_hand_presence_confidence": min_hand_presence_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        if detector_options is not None:
            options.update(detector_options)
        factory = detector_factory or _optional_class_factory(
            module_name,
            detector_class_name,
            options,
        )
        self.lazy_detector = LazyDetector(factory, name="hands")
        super().__init__(self.lazy_detector)

    def _run(self, image: np.ndarray) -> object:
        raw = self.lazy_detector.detect(image)
        if isinstance(raw, Mapping) and (
            raw.get("status") in {
                DetectionStatus.UNAVAILABLE.value,
                DetectionStatus.FAILED.value,
            }
            or raw.get("success") is False
        ):
            return raw
        if not isinstance(raw, Mapping):
            return raw
        payload = dict(raw)
        payload.setdefault("coordinate_system", "normalized")
        payload.setdefault("handedness_convention", "subject")
        payload.setdefault("labels_are_subject_relative", True)
        return payload


LazySegmentationDetectorAdapter = MediaPipeSegmentationAdapter
LazyHandDetectorAdapter = MediaPipeHandDetectorAdapter


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
    normalised_results = dict(results)
    for name, result in results.items():
        if not isinstance(result, DetectorResult):
            raise TypeError(f"{name} result must be a DetectorResult")
        diagnostics.extend(result.diagnostics)
        if result.available:
            try:
                detections = merge_body_detections(detections, result.detections)
            except ValueError as error:
                message = f"could not merge {name} detections: {error}"
                diagnostics.append(message)
                normalised_results[name] = DetectorResult(
                    detector=name,
                    status=DetectionStatus.FAILED,
                    diagnostics=(message,),
                )
    return BodyDetectionResult(
        detections=detections,
        detector_results=normalised_results,
        diagnostics=tuple(diagnostics),
    )


def _adapt_detector(
    name: str,
    detector: object | None,
) -> object:
    if detector is None:
        return UnavailableDetector(name)
    if isinstance(
        detector,
        (
            _NormalisingAdapter,
            PersonFallbackAdapter,
            UnavailableDetector,
        ),
    ):
        return detector
    if name == "pose":
        return PoseDetectorAdapter(detector)  # type: ignore[arg-type]
    if name == "head":
        return HeadDetectorAdapter(detector)  # type: ignore[arg-type]
    if name == "segmentation":
        return SegmentationDetectorAdapter(detector)  # type: ignore[arg-type]
    if name == "hands":
        return HandDetectorAdapter(detector)  # type: ignore[arg-type]
    raise ValueError(f"unsupported body detector: {name}")


class BodyDetectionOrchestrator:
    """Run configured body detectors and retain partial results."""

    def __init__(
        self,
        *,
        configuration: BodyDetectionConfig | None = None,
        pose_detector: object | None = None,
        head_detector: object | None = None,
        segmentation_detector: object | None = None,
        hand_detector: object | None = None,
        use_person_fallback: bool = True,
    ) -> None:
        if configuration is not None and not isinstance(
            configuration,
            BodyDetectionConfig,
        ):
            raise TypeError("body detection configuration is invalid")
        configuration = configuration or BodyDetectionConfig()
        self.detectors = {
            "pose": _adapt_detector(
                "pose",
                pose_detector if pose_detector is not None else configuration.pose_detector,
            ),
            "head": _adapt_detector(
                "head",
                head_detector if head_detector is not None else configuration.head_detector,
            ),
            "segmentation": _adapt_detector(
                "segmentation",
                (
                    segmentation_detector
                    if segmentation_detector is not None
                    else configuration.segmentation_detector
                ),
            ),
            "hands": _adapt_detector(
                "hands",
                hand_detector if hand_detector is not None else configuration.hand_detector,
            ),
        }
        if use_person_fallback:
            self.detectors["person_fallback"] = PersonFallbackAdapter()

    @staticmethod
    def _load_image(image: str | Path | np.ndarray) -> tuple[np.ndarray | None, str | None]:
        if isinstance(image, np.ndarray):
            if image.size == 0 or image.ndim not in {2, 3}:
                return None, "image must be a non-empty two- or three-dimensional array"
            if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
                return None, "image must have one, three, or four channels"
            return image, None
        try:
            image_array = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        except (TypeError, ValueError) as error:
            return None, f"could not read image: {error}"
        if image_array is None or image_array.size == 0:
            return None, f"could not read image: {image}"
        return image_array, None

    def detect(self, image: str | Path | np.ndarray) -> BodyDetectionResult:
        """Detect a body from an image path or an already-loaded image array."""
        image_array, error = self._load_image(image)
        if image_array is None:
            message = error or "could not read image"
            input_result = DetectorResult(
                detector="input",
                status=DetectionStatus.FAILED,
                diagnostics=(message,),
            )
            return BodyDetectionResult(
                detector_results={"input": input_result},
                diagnostics=(message,),
            )

        results: dict[str, DetectorResult] = {}
        for name, detector in self.detectors.items():
            try:
                result = detector.detect(image_array)  # type: ignore[attr-defined]
            except Exception as error:
                result = DetectorResult(
                    detector=name,
                    status=DetectionStatus.FAILED,
                    diagnostics=(f"{name} failed: {error}",),
                )
            if not isinstance(result, DetectorResult):
                result = DetectorResult(
                    detector=name,
                    status=DetectionStatus.FAILED,
                    diagnostics=(
                        f"{name} returned an invalid detector result",
                    ),
                )
            results[name] = result
        return build_body_detection_result(results)

    def detect_all(self, image: str | Path | np.ndarray) -> BodyDetectionResult:
        """Compatibility name for callers of the legacy Layer 3 interface."""
        return self.detect(image)


UnifiedBodyDetector = BodyDetectionOrchestrator
HumanBodyUnderstanding = BodyDetectionOrchestrator


# American spelling is kept as a small compatibility alias for callers.
normalize_pose_keypoints = normalise_pose_keypoints
normalize_head_detection = normalise_head_detection
BodyPoseDetector = PoseDetectorAdapter
HeadDetector = HeadDetectorAdapter