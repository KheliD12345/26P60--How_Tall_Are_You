"""Compare frame-tag rectification with the independent stick-tag scale.

The experimental runner uses the same ViTPose heel and VGGHeads head-top
landmarks as the production pipeline. Four frame markers define the primary
homography when available. A separate scale from stick markers 4 and 5 is
reported as a calibration check.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2

from height_estimation.body_detection import (
    BodyDetectionConfig,
    BodyDetectionOrchestrator,
    ViTPoseDetectorAdapter,
    VGGHeadsDetectorAdapter,
)
from height_estimation.frame_stick import (
    DEFAULT_MARKER_TYPE,
    MARKER_CONFIGS,
    MarkerConfig,
    detect_subject,
    detect_markers,
    estimate_height,
    point_to_dict,
    require_markers,
    write_overlay,
)
from height_estimation.models import MarkerLayout


SCRIPT_DIR = Path(__file__).resolve().parent
FRAME_LAYOUT_PATH = SCRIPT_DIR.parent / "configs" / "apriltag_hybrid_layout.json"
IMAGE_NAMES = (
    "ApriltagHybrid1.jpg",
    "ApriltagHybrid2.jpg",
)


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