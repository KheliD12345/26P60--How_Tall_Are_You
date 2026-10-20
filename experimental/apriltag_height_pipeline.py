"""Compare frame-tag rectification with the independent stick-tag scale.

The experimental runner uses the same ViTPose heel and VGGHeads head-top
landmarks as the production pipeline. Four frame markers define the primary
homography when available. A separate scale from stick markers 4 and 5 is
reported as a calibration check.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, replace
from math import atan2, degrees, hypot, isfinite
from pathlib import Path

import cv2
import numpy as np

from height_estimation.advanced_models import BodyDetections
from height_estimation.aruco import detect_markers_for_dictionary
from height_estimation.body_detection import (
    BodyDetectionConfig,
    BodyDetectionOrchestrator,
    ViTPoseDetectorAdapter,
    VGGHeadsDetectorAdapter,
)
from height_estimation.geometry import estimate_homography, transform_point
from height_estimation.models import DetectedMarker, MarkerLayout


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MARKER_TYPE = "apriltag"
FRAME_LAYOUT_PATH = SCRIPT_DIR.parent / "configs" / "apriltag_hybrid_layout.json"
IMAGE_NAMES = (
    "ApriltagHybrid1.jpg",
    "ApriltagHybrid2.jpg",
)


@dataclass(frozen=True)
class MarkerConfig:
    marker_type: str
    dictionary_name: str
    display_name: str
    reference_ids: tuple[int, int]
    marker_size_cm: float


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


def _point_dict(point: tuple[float, float]) -> dict[str, float]:
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
) -> tuple[SubjectEndpoints, object]:
    body_result = body_detector.detect(image)
    detections: BodyDetections = body_result.detections
    head = detections.head_top or detections.keypoints.get("head_top")
    heels = detections.heel_landmarks()
    if head is None or not heels:
        details = "; ".join(body_result.diagnostics)
        reason = "shared head-top and heel landmarks were not detected"
        raise ValueError(f"{reason}: {details}" if details else reason)

    height, width = image.shape[:2]
    head_top = head.to_pixel(width, height)
    heel_points = tuple(heel.to_pixel(width, height) for heel in heels)
    return (
        SubjectEndpoints(
            head_top=head_top,
            heel_points=heel_points,
            confidence=detections.head_confidence,
        ),
        body_result,
    )


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
) -> dict[str, object]:
    frame_height: float | None = None
    frame_details: dict[str, object] | None = None
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
    stick_check: dict[str, object] | None = None
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

    height_cm = frame_height if frame_height is not None else stick_check["height_cm"]
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
        "head_top": _point_dict(subject.head_top),
        "feet_bottom": _point_dict(selected_heel),
        "frame_homography": frame_details,
        "stick_scale_check": stick_check,
        "endpoint_method": "shared VGGHeads head-top and ViTPose heel landmarks",
    }


def write_overlay(
    image: np.ndarray,
    frame_markers: dict[int, MarkerDetection],
    stick_markers: tuple[MarkerDetection, MarkerDetection] | None,
    subject: SubjectEndpoints,
    height: dict[str, object],
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
    feet = tuple(round(value) for value in height["feet_bottom"].values())
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
    stick_markers = (
        tuple(stick_detections[marker_id] for marker_id in config.reference_ids)
        if all(marker_id in stick_detections for marker_id in config.reference_ids)
        else None
    )
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
                name: detector_result.status.value
                for name, detector_result in body_result.detector_results.items()
            },
            "subject_confidence": round(subject.confidence, 4),
        },
        "frame_markers": [
            {
                "id": marker.id,
                "center_px": _point_dict(marker.center),
                "area_px": round(marker.area_px, 2),
                "corners_px": [_point_dict(point) for point in marker.corners],
            }
            for marker in frame_markers.values()
        ],
        "stick_markers": [
            {
                "id": marker.id,
                "center_px": _point_dict(marker.center),
                "area_px": round(marker.area_px, 2),
                "corners_px": [_point_dict(point) for point in marker.corners],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marker-type",
        "--marker-family",
        dest="marker_type",
        choices=tuple(MARKER_CONFIGS),
        default=DEFAULT_MARKER_TYPE,
        help=(
            "fiducial marker family to detect; aruco uses DICT_4X4_50 and "
            "apriltag uses DICT_APRILTAG_36h11"
        ),
    )
    parser.add_argument(
        "--reference-ids",
        nargs=2,
        type=int,
        metavar=("ID_A", "ID_B"),
        help=(
            "override the two marker IDs used for the 81 cm scale; "
            "AprilTag defaults to 4 5 and ArUco defaults to 0 1"
        ),
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="image paths; defaults to the two JPEGs in this folder",
    )
    parser.add_argument(
        "--frame-layout",
        type=Path,
        default=FRAME_LAYOUT_PATH,
        help="four-marker frame layout JSON used for primary rectification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "directory for JSON results and annotated images; defaults to "
            "experimental/results/<marker-type>"
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue processing later images if one image fails",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    if path.is_file() or path.is_absolute():
        return path
    local_path = SCRIPT_DIR / path
    return local_path if local_path.is_file() else path


def main() -> int:
    args = parse_args()
    config = MARKER_CONFIGS[args.marker_type]
    if args.reference_ids is not None:
        reference_ids = tuple(args.reference_ids)
        if reference_ids[0] == reference_ids[1]:
            raise SystemExit("reference IDs must be different")
        config = replace(config, reference_ids=reference_ids)

    requested_images = args.images or [
        SCRIPT_DIR.parent / "inputs" / name for name in IMAGE_NAMES
    ]
    image_paths = tuple(_resolve_path(path) for path in requested_images)
    missing_images = [path.name for path in image_paths if not path.is_file()]
    if missing_images:
        raise SystemExit(f"missing required test image(s): {', '.join(missing_images)}")

    try:
        frame_layout = MarkerLayout.from_json(_resolve_path(args.frame_layout))
    except (OSError, ValueError) as error:
        raise SystemExit(f"could not load frame layout: {error}") from error

    body_detector = BodyDetectionOrchestrator(
        configuration=BodyDetectionConfig(
            pose_detector=ViTPoseDetectorAdapter(),
            head_detector=VGGHeadsDetectorAdapter(),
        ),
        use_person_fallback=False,
    )
    output_dir = args.output_dir or SCRIPT_DIR / "results" / config.marker_type
    results = []
    failures = []
    for path in image_paths:
        try:
            results.append(
                process_image(body_detector, path, output_dir, config, frame_layout)
            )
        except (OSError, ValueError, RuntimeError) as error:
            failures.append({"image": path.name, "error": str(error)})
            if not args.continue_on_error:
                raise
    summary_path = output_dir / "summary.json"
    summary = {"results": results, "failures": failures}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for result in results:
        estimate = result["estimate"]
        stick_check = estimate["stick_scale_check"]
        stick_text = (
            f"stick check {stick_check['height_cm']:.2f} cm"
            if stick_check is not None
            else "stick check unavailable"
        )
        print(
            f"{result['image']}: "
            f"{estimate['height_cm']:.2f} cm via {estimate['method']}; "
            f"{stick_text}"
        )
    print(
        f"Stick markers: {config.marker_type} ({config.dictionary_name}), "
        f"frame markers: {frame_layout.dictionary}"
    )
    if failures:
        print(f"Failed images: {len(failures)}")
        for failure in failures:
            print(f"{failure['image']}: FAILED: {failure['error']}")
    print(f"Wrote results to {output_dir}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())