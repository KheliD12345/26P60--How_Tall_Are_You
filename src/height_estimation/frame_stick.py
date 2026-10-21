"""Height estimation from frame homography and stick marker scale."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from .advanced_models import BodyDetections, Landmark
from .aruco import detect_markers_for_dictionary
from .body_detection import (
    BodyDetectionOrchestrator,
    BodyDetectionResult,
    DetectionStatus,
)
from .geometry import estimate_homography, transform_point
from .models import DetectedMarker, MarkerLayout


@dataclass(frozen=True)
class MarkerConfig:
    marker_type: str
    dictionary_name: str
    display_name: str
    reference_ids: tuple[int, int]
    marker_size_cm: float


@dataclass(frozen=True)
class MarkerDetection:
    id: int
    corners: tuple[tuple[float, float], ...]
    center: tuple[float, float]
    area_px: float


@dataclass(frozen=True)
class SubjectEndpoints:
    head_top: tuple[float, float]
    heel_points: tuple[tuple[float, float], ...]
    confidence: float


class PixelPoint(TypedDict):
    x_px: float
    y_px: float


class MetricPoint(TypedDict):
    x: float
    y: float


class FrameHomographyEstimate(TypedDict):
    height_cm: float
    reprojection_error_cm: float
    head_top_cm: MetricPoint
    feet_bottom_cm: MetricPoint


class StickScaleEstimate(TypedDict):
    reference_distance_cm: float
    reference_distance_px: float
    reference_delta_px: MetricPoint
    reference_axis_angle_degrees: float
    cm_per_pixel: float
    height_cm: float
    difference_from_frame_homography_cm: float | None


class FrameStickEstimate(TypedDict):
    method: str
    height_cm: float
    subject_height_pixels: float
    head_top: PixelPoint
    feet_bottom: PixelPoint
    frame_homography: FrameHomographyEstimate | None
    stick_scale_check: StickScaleEstimate | None
    endpoint_method: str


DEFAULT_MARKER_TYPE = "apriltag"
MARKER_CONFIGS = {
    "aruco": MarkerConfig(
        marker_type="aruco",
        dictionary_name="DICT_4X4_50",
        display_name="ArUco",
        reference_ids=(0, 1),
        marker_size_cm=18.0,
    ),
    "apriltag": MarkerConfig(
        marker_type="apriltag",
        dictionary_name="DICT_APRILTAG_36h11",
        display_name="AprilTag",
        reference_ids=(4, 5),
        marker_size_cm=18.0,
    ),
}


def point_to_dict(point: tuple[float, float]) -> PixelPoint:
    return {"x_px": round(point[0], 3), "y_px": round(point[1], 3)}


def detect_markers(
    image: np.ndarray,
    dictionary_name: str,
    marker_ids: tuple[int, ...],
) -> dict[int, MarkerDetection]:
    markers = detect_markers_for_dictionary(
        image,
        dictionary_name,
        required_ids=marker_ids,
    )
    return {
        marker.id: MarkerDetection(
            id=marker.id,
            corners=marker.corners,
            center=(marker.center_x, marker.center_y),
            area_px=marker.area_px,
        )
        for marker in markers
        if marker.id in marker_ids
    }


def require_markers(
    detections: dict[int, MarkerDetection],
    marker_ids: tuple[int, ...],
    label: str,
) -> tuple[MarkerDetection, ...]:
    missing = [marker_id for marker_id in marker_ids if marker_id not in detections]
    if missing:
        raise ValueError(
            f"missing required {label} IDs: {', '.join(map(str, missing))}"
        )
    return tuple(detections[marker_id] for marker_id in marker_ids)


def detect_subject(
    body_detector: BodyDetectionOrchestrator,
    image: np.ndarray,
) -> tuple[SubjectEndpoints, BodyDetectionResult]:
    body_result = body_detector.detect(image)
    detections: BodyDetections = body_result.detections
    head = detections.head_top
    if head is None:
        head = detections.keypoints.get("head_top")
    heels = detections.heel_landmarks()
    if head is None or not heels:
        details = "; ".join(body_result.diagnostics)
        reason = "shared head-top and heel landmarks were not detected"
        raise ValueError(f"{reason}: {details}" if details else reason)

    height, width = image.shape[:2]
    head_top = Landmark.from_value(head).to_pixel(width, height)
    heel_points = tuple(heel.to_pixel(width, height) for heel in heels)
    return (
        SubjectEndpoints(
            head_top=head_top,
            heel_points=heel_points,
            confidence=detections.head_confidence,
        ),
        body_result,
    )


def write_overlay(
    image: np.ndarray,
    frame_markers: dict[int, MarkerDetection],
    stick_markers: tuple[MarkerDetection, MarkerDetection] | None,
    subject: SubjectEndpoints,
    height: FrameStickEstimate,
    config: MarkerConfig,
    output_path: Path,
) -> None:
    overlay = image.copy()
    for marker in frame_markers.values():
        points = np.asarray(marker.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], True, (255, 160, 0), 3)
        center = tuple(round(value) for value in marker.center)
        cv2.circle(overlay, center, 7, (255, 160, 0), -1)
        cv2.putText(
            overlay,
            f"Frame {marker.id}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 90, 0),
            2,
            cv2.LINE_AA,
        )

    if stick_markers is not None:
        for marker in stick_markers:
            points = np.asarray(marker.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [points], True, (0, 180, 0), 3)
            center = tuple(round(value) for value in marker.center)
            cv2.circle(overlay, center, 7, (0, 180, 0), -1)
            cv2.putText(
                overlay,
                f"{config.display_name} {marker.id}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 120, 0),
                2,
                cv2.LINE_AA,
            )
        first_center = tuple(round(value) for value in stick_markers[0].center)
        second_center = tuple(round(value) for value in stick_markers[1].center)
        cv2.line(overlay, first_center, second_center, (0, 180, 0), 3)

    head = tuple(round(value) for value in subject.head_top)
    feet_bottom = height["feet_bottom"]
    feet = (round(feet_bottom["x_px"]), round(feet_bottom["y_px"]))
    cv2.circle(overlay, head, 8, (0, 0, 255), -1)
    cv2.circle(overlay, feet, 8, (0, 0, 255), -1)
    cv2.line(overlay, head, feet, (0, 0, 255), 3)

    stick_check = height["stick_scale_check"]
    label = (
        f"height {height['height_cm']:.2f} cm | "
        + (
            f"stick check {stick_check['height_cm']:.2f} cm"
            if stick_check is not None
            else "stick check unavailable"
        )
    )
    cv2.rectangle(overlay, (20, 20), (900, 65), (255, 255, 255), -1)
    cv2.putText(
        overlay,
        label,
        (30, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"could not write overlay: {output_path}")


def _detected_marker(marker: MarkerDetection) -> DetectedMarker:
    return DetectedMarker(
        id=marker.id,
        corners=marker.corners,
        center_x=marker.center[0],
        center_y=marker.center[1],
        area_px=marker.area_px,
    )


def estimate_height(
    frame_markers: dict[int, MarkerDetection],
    stick_markers: tuple[MarkerDetection, MarkerDetection] | None,
    subject: SubjectEndpoints,
    frame_layout: MarkerLayout,
) -> FrameStickEstimate:
    frame_height: float | None = None
    frame_details: FrameHomographyEstimate | None = None
    selected_heel = max(subject.heel_points, key=lambda point: point[1])
    if all(marker_id in frame_markers for marker_id in frame_layout.marker_ids):
        detected = tuple(
            _detected_marker(frame_markers[marker_id])
            for marker_id in frame_layout.marker_ids
        )
        homography = estimate_homography(detected, frame_layout)
        head_metric = transform_point(subject.head_top, homography.matrix)
        transformed_heels = tuple(
            transform_point(point, homography.matrix)
            for point in subject.heel_points
        )
        heel_index = max(
            range(len(transformed_heels)),
            key=lambda index: transformed_heels[index][1],
        )
        selected_heel = subject.heel_points[heel_index]
        frame_height = abs(transformed_heels[heel_index][1] - head_metric[1])
        frame_details = {
            "height_cm": round(frame_height, 2),
            "reprojection_error_cm": round(homography.reprojection_error_cm, 4),
            "head_top_cm": {
                "x": round(head_metric[0], 3),
                "y": round(head_metric[1], 3),
            },
            "feet_bottom_cm": {
                "x": round(transformed_heels[heel_index][0], 3),
                "y": round(transformed_heels[heel_index][1], 3),
            },
        }

    height_pixels = abs(selected_heel[1] - subject.head_top[1])
    stick_check: StickScaleEstimate | None = None
    if stick_markers is not None:
        first_center, second_center = stick_markers[0].center, stick_markers[1].center
        reference_dx = second_center[0] - first_center[0]
        reference_dy = second_center[1] - first_center[1]
        reference_pixels = hypot(reference_dx, reference_dy)
        if reference_pixels <= 0.0:
            raise ValueError("reference marker centres must be separated in the image")

        reference_distance_cm = frame_layout.stick_reference_distance_cm
        cm_per_pixel = reference_distance_cm / reference_pixels
        stick_height_cm = height_pixels * cm_per_pixel
        stick_check = {
            "reference_distance_cm": reference_distance_cm,
            "reference_distance_px": round(reference_pixels, 3),
            "reference_delta_px": {
                "x": round(reference_dx, 3),
                "y": round(reference_dy, 3),
            },
            "reference_axis_angle_degrees": round(
                degrees(atan2(reference_dy, reference_dx)),
                3,
            ),
            "cm_per_pixel": round(cm_per_pixel, 6),
            "height_cm": round(stick_height_cm, 2),
            "difference_from_frame_homography_cm": (
                round(stick_height_cm - frame_height, 2)
                if frame_height is not None
                else None
            ),
        }
    elif frame_height is None:
        raise ValueError(
            "frame homography is unavailable and both stick markers are required "
            "for fallback"
        )

    # Frame homography takes priority; stick distance is the fallback.
    if frame_height is None:
        if stick_check is None:
            raise ValueError("neither frame nor stick markers produced an estimate")
        height_cm = stick_check["height_cm"]
    else:
        height_cm = frame_height
    if not isfinite(height_cm) or height_cm <= 0:
        raise ValueError("height estimate must be finite and greater than zero")
    return {
        "method": (
            "four_frame_marker_homography"
            if frame_height is not None
            else "stick_marker_scalar_fallback"
        ),
        "height_cm": round(height_cm, 2),
        "subject_height_pixels": round(height_pixels, 3),
        "head_top": point_to_dict(subject.head_top),
        "feet_bottom": point_to_dict(selected_heel),
        "frame_homography": frame_details,
        "stick_scale_check": stick_check,
        "endpoint_method": "shared VGGHeads head-top and ViTPose heel landmarks",
    }


def process_image(
    body_detector: BodyDetectionOrchestrator,
    image_path: Path,
    output_dir: Path,
    config: MarkerConfig,
    frame_layout: MarkerLayout,
) -> dict[str, object]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"could not read image: {image_path}")
    frame_markers = detect_markers(
        image,
        frame_layout.dictionary,
        frame_layout.marker_ids,
    )
    stick_detections = detect_markers(
        image,
        config.dictionary_name,
        config.reference_ids,
    )
    stick_markers: tuple[MarkerDetection, MarkerDetection] | None = None
    if all(marker_id in stick_detections for marker_id in config.reference_ids):
        first_id, second_id = config.reference_ids
        stick_markers = (stick_detections[first_id], stick_detections[second_id])
    subject, body_result = detect_subject(body_detector, image)
    frame_is_complete = all(
        marker_id in frame_markers for marker_id in frame_layout.marker_ids
    )
    if not frame_is_complete and stick_markers is None:
        require_markers(stick_detections, config.reference_ids, config.display_name)
    height = estimate_height(frame_markers, stick_markers, subject, frame_layout)
    result = {
        "image": image_path.name,
        "image_size_px": {"width": image.shape[1], "height": image.shape[0]},
        "detector": {
            "frame_marker_family": frame_layout.dictionary,
            "frame_marker_ids": list(frame_layout.marker_ids),
            "stick_marker_family": config.dictionary_name,
            "stick_marker_ids": list(config.reference_ids),
            "body_detectors": {
                name: DetectionStatus(detector_result.status).value
                for name, detector_result in body_result.detector_results.items()
            },
            "subject_confidence": round(subject.confidence, 4),
        },
        "frame_markers": [
            {
                "id": marker.id,
                "center_px": point_to_dict(marker.center),
                "area_px": round(marker.area_px, 2),
                "corners_px": [point_to_dict(point) for point in marker.corners],
            }
            for marker in frame_markers.values()
        ],
        "stick_markers": [
            {
                "id": marker.id,
                "center_px": point_to_dict(marker.center),
                "area_px": round(marker.area_px, 2),
                "corners_px": [point_to_dict(point) for point in marker.corners],
            }
            for marker in stick_detections.values()
        ],
        "estimate": height,
        "body_detection_diagnostics": list(body_result.diagnostics),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_overlay(
        image,
        frame_markers,
        stick_markers,
        subject,
        height,
        config,
        output_dir / f"{image_path.stem}_overlay.jpg",
    )
    return result
