"""Estimate subject height from two AprilTags and direct pose endpoints.

The default run is intentionally limited to the two JPEGs in this folder.
It uses the 81 cm centre-to-centre distance between AprilTag IDs 0 and 1 as
the local metric scale. Subject endpoints are the top and bottom of the
selected pose person box. No head-height ratio or anthropometric estimate is
used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import degrees, hypot
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DISTANCE_CM = 81.0
REFERENCE_IDS = (0, 1)
POSE_MODEL_NAME = "yolo11n-pose.pt"
IMAGE_NAMES = (
    "WhatsApp Image 2026-10-05 at 5.12.58 PM (1).jpeg",
    "WhatsApp Image 2026-10-05 at 5.12.58 PM.jpeg",
)


@dataclass(frozen=True)
class MarkerDetection:
    id: int
    corners: tuple[tuple[float, float], ...]
    center: tuple[float, float]
    area_px: float


@dataclass(frozen=True)
class SubjectEndpoints:
    box: tuple[float, float, float, float]
    head_top: tuple[float, float]
    feet_bottom: tuple[float, float]
    confidence: float


def _point_dict(point: tuple[float, float]) -> dict[str, float]:
    return {"x_px": round(point[0], 3), "y_px": round(point[1], 3)}


def detect_reference_markers(image: np.ndarray) -> tuple[MarkerDetection, ...]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        raise ValueError("no DICT_APRILTAG_36h11 markers were detected")

    detections: dict[int, MarkerDetection] = {}
    for raw_corners, raw_id in zip(corners, ids.flatten().tolist()):
        points = np.asarray(raw_corners, dtype=np.float64).reshape(4, 2)
        marker_id = int(raw_id)
        if marker_id in detections:
            raise ValueError(f"duplicate AprilTag ID detected: {marker_id}")
        detections[marker_id] = MarkerDetection(
            id=marker_id,
            corners=tuple((float(x), float(y)) for x, y in points),
            center=(float(points[:, 0].mean()), float(points[:, 1].mean())),
            area_px=float(abs(cv2.contourArea(points.astype(np.float32)))),
        )

    missing = [marker_id for marker_id in REFERENCE_IDS if marker_id not in detections]
    if missing:
        raise ValueError(f"missing required AprilTag IDs: {', '.join(map(str, missing))}")
    return tuple(detections[marker_id] for marker_id in REFERENCE_IDS)


def detect_subject(model: YOLO, image: np.ndarray) -> SubjectEndpoints:
    result = model.predict(
        source=image,
        classes=[0],
        conf=0.25,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError("no person was detected")

    candidates: list[tuple[float, float, tuple[float, float, float, float]]] = []
    for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        x1, y1, x2, y2 = (float(value) for value in box)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        candidates.append((area, float(confidence), (x1, y1, x2, y2)))
    _, confidence, box = max(candidates, key=lambda candidate: candidate[0] * candidate[1])
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0
    return SubjectEndpoints(
        box=box,
        head_top=(center_x, y1),
        feet_bottom=(center_x, y2),
        confidence=confidence,
    )


def estimate_height(
    markers: tuple[MarkerDetection, MarkerDetection],
    subject: SubjectEndpoints,
) -> dict[str, object]:
    first_center, second_center = markers[0].center, markers[1].center
    reference_dx = second_center[0] - first_center[0]
    reference_dy = second_center[1] - first_center[1]
    reference_pixels = hypot(reference_dx, reference_dy)
    if reference_pixels <= 0.0:
        raise ValueError("AprilTag centres must be separated in the image")

    cm_per_pixel = REFERENCE_DISTANCE_CM / reference_pixels
    height_pixels = abs(subject.feet_bottom[1] - subject.head_top[1])
    height_cm = height_pixels * cm_per_pixel
    return {
        "reference_distance_cm": REFERENCE_DISTANCE_CM,
        "reference_distance_px": round(reference_pixels, 3),
        "reference_delta_px": {
            "x": round(reference_dx, 3),
            "y": round(reference_dy, 3),
        },
        "reference_axis_angle_degrees": round(degrees(np.arctan2(reference_dy, reference_dx)), 3),
        "cm_per_pixel": round(cm_per_pixel, 6),
        "subject_height_pixels": round(height_pixels, 3),
        "height_cm": round(height_cm, 2),
        "head_top": _point_dict(subject.head_top),
        "feet_bottom": _point_dict(subject.feet_bottom),
        "endpoint_method": "direct top and bottom of selected YOLO pose person box",
    }


def write_overlay(
    image: np.ndarray,
    markers: tuple[MarkerDetection, MarkerDetection],
    subject: SubjectEndpoints,
    height: dict[str, object],
    output_path: Path,
) -> None:
    overlay = image.copy()
    for marker in markers:
        points = np.asarray(marker.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], True, (0, 180, 0), 3)
        center = tuple(round(value) for value in marker.center)
        cv2.circle(overlay, center, 7, (0, 180, 0), -1)
        cv2.putText(
            overlay,
            f"AprilTag {marker.id}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 120, 0),
            2,
            cv2.LINE_AA,
        )

    first_center = tuple(round(value) for value in markers[0].center)
    second_center = tuple(round(value) for value in markers[1].center)
    cv2.line(overlay, first_center, second_center, (0, 180, 0), 3)

    x1, y1, x2, y2 = (round(value) for value in subject.box)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 120, 0), 3)
    head = tuple(round(value) for value in subject.head_top)
    feet = tuple(round(value) for value in subject.feet_bottom)
    cv2.circle(overlay, head, 8, (0, 0, 255), -1)
    cv2.circle(overlay, feet, 8, (0, 0, 255), -1)
    cv2.line(overlay, head, feet, (0, 0, 255), 3)

    label = f"height {height['height_cm']:.2f} cm | scale {height['cm_per_pixel']:.5f} cm/px"
    cv2.rectangle(overlay, (20, 20), (750, 65), (255, 255, 255), -1)
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


def process_image(model: YOLO, image_path: Path, output_dir: Path) -> dict[str, object]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"could not read image: {image_path}")
    markers = detect_reference_markers(image)
    subject = detect_subject(model, image)
    height = estimate_height(markers, subject)
    result = {
        "image": image_path.name,
        "image_size_px": {"width": image.shape[1], "height": image.shape[0]},
        "detector": {
            "marker_family": "DICT_APRILTAG_36h11",
            "required_ids": list(REFERENCE_IDS),
            "subject_model": POSE_MODEL_NAME,
            "subject_confidence": round(subject.confidence, 4),
        },
        "markers": [
            {
                "id": marker.id,
                "center_px": _point_dict(marker.center),
                "area_px": round(marker.area_px, 2),
                "corners_px": [_point_dict(point) for point in marker.corners],
            }
            for marker in markers
        ],
        "subject_box_px": {
            "x1": round(subject.box[0], 3),
            "y1": round(subject.box[1], 3),
            "x2": round(subject.box[2], 3),
            "y2": round(subject.box[3], 3),
        },
        "estimate": height,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_overlay(
        image,
        markers,
        subject,
        height,
        output_dir / f"{image_path.stem}_overlay.jpg",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=SCRIPT_DIR / POSE_MODEL_NAME,
        help="path to a YOLO pose model weight",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="directory for JSON results and annotated images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = tuple(SCRIPT_DIR / name for name in IMAGE_NAMES)
    missing_images = [path.name for path in image_paths if not path.is_file()]
    if missing_images:
        raise SystemExit(f"missing required test image(s): {', '.join(missing_images)}")
    if not args.model.is_file():
        raise SystemExit(
            f"missing pose model: {args.model}. "
            "Place yolo11n-pose.pt in experimental or pass --model."
        )

    model = YOLO(str(args.model))
    results = [process_image(model, path, args.output_dir) for path in image_paths]
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        estimate = result["estimate"]
        print(
            f"{result['image']}: "
            f"{estimate['height_cm']:.2f} cm "
            f"({estimate['subject_height_pixels']:.2f} px, "
            f"{estimate['cm_per_pixel']:.6f} cm/px)"
        )
    print(f"Wrote results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())