import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .body_detection import model_backed_body_detection_config
from .calibration import calibrate_image
from .models import MarkerLayout
from .pipeline import (
    MeasurementPipeline,
    PipelineConfig,
    PipelineFailure,
    PipelineInput,
    PipelineStage,
)
from .visualization import write_calibration_overlay


DEFAULT_LAYOUT = Path(__file__).resolve().parents[2] / "configs" / "marker_layout.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a known ArUco marker layout for height estimation."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--layout",
        type=Path,
        default=DEFAULT_LAYOUT,
        help="path to a marker layout JSON file",
    )
    parser.add_argument("image", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, help="path for detection JSON")
    parser.add_argument("--overlay", type=Path, help="path for calibration overlay")
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        help="optional camera calibration JSON for undistortion",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="run the complete measurement pipeline",
    )
    parser.add_argument(
        "--quality-policy",
        choices=("fail", "continue"),
        default="fail",
        help="stop on quality failure or continue with a warning",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="confidence level for the fused uncertainty interval",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="process all positional images and write one result per image",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for batch pipeline result JSON files",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue batch processing after an image fails",
    )
    parser.add_argument(
        "--no-person-fallback",
        action="store_true",
        help="disable the optional HOG/GrabCut person fallback",
    )
    parser.add_argument(
        "--model-detectors",
        action="store_true",
        help="enable optional pose, head, segmentation, and hand detectors",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = MarkerLayout.from_json(args.layout)
    if args.pipeline:
        return _run_pipeline_mode(args, layout)
    if args.batch or args.output_dir is not None:
        build_parser().error("--batch and --output-dir require --pipeline")
    if len(args.image) > 1:
        build_parser().error("multiple images require --batch")
    if args.image:
        return _run_calibration_mode(args, layout)

    print(f"Loaded {len(layout.markers)} markers from {args.layout}")
    print(f"Dictionary: {layout.dictionary}")
    print(f"Marker IDs: {', '.join(map(str, layout.marker_ids))}")
    return 0


def _run_calibration_mode(args: argparse.Namespace, layout: MarkerLayout) -> int:
    image = args.image[0]
    try:
        if args.camera_calibration is None:
            calibration = calibrate_image(image, layout)
        else:
            calibration = calibrate_image(
                image,
                layout,
                camera_calibration_path=args.camera_calibration,
            )
        if args.overlay is not None:
            write_calibration_overlay(image, calibration, args.overlay)
    except (OSError, ValueError) as error:
        build_parser().error(str(error))
    result = {"image": str(image), **calibration.to_dict()}
    output = json.dumps(result, indent=2)
    if args.output is None:
        print(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(
            f"Wrote {len(calibration.markers)} markers, "
            f"{len(calibration.geometry)} pairs, "
            f"scale {calibration.cm_per_pixel:.6f} cm/pixel, "
            f"reprojection error "
            f"{calibration.homography.reprojection_error_cm:.6f} cm "
            f"to {args.output}"
        )
    if args.overlay is not None:
        print(f"Wrote calibration overlay to {args.overlay}")
    return 0


def _run_pipeline_mode(args: argparse.Namespace, layout: MarkerLayout) -> int:
    images = args.image or []
    if not images:
        build_parser().error("an image path is required in pipeline mode")
    if args.batch and len(images) < 1:
        build_parser().error("--batch requires at least one image path")
    if args.batch and args.output is not None:
        build_parser().error("--output cannot be used with --batch")
    if len(images) > 1 and not args.batch:
        build_parser().error("multiple images require --batch")
    if args.output_dir is not None and not args.batch:
        build_parser().error("--output-dir requires --batch")

    output_dir = args.output_dir
    if args.batch and output_dir is None:
        output_dir = Path("outputs")
    failures = 0
    for image in images:
        try:
            output_path = args.output
            if args.batch and output_dir is not None:
                output_path = output_dir / f"{image.stem}.json"
            config = PipelineConfig(
                quality_policy=args.quality_policy,
                confidence_level=args.confidence_level,
                use_person_fallback=not args.no_person_fallback,
                output_path=output_path,
                body_detection=(
                    model_backed_body_detection_config()
                    if args.model_detectors
                    else None
                ),
            )
            result = MeasurementPipeline(config=config).run(
                PipelineInput(
                    image_path=image,
                    layout=layout,
                    camera_calibration_path=args.camera_calibration,
                )
            )
            if args.overlay is not None:
                calibration_stage = result.stages.get(PipelineStage.CALIBRATION)
                calibration = (
                    None
                    if calibration_stage is None
                    else calibration_stage.value
                )
                if calibration is None:
                    raise PipelineFailure(
                        "output_write_failed",
                        "calibration result was unavailable for overlay",
                        stage=PipelineStage.PERSISTENCE,
                    )
                write_calibration_overlay(image, calibration, args.overlay)
            if output_path is None:
                print(result.measurement_result.to_json(indent=2))
            else:
                print(f"Wrote measurement result to {output_path}")
            if args.overlay is not None:
                print(f"Wrote calibration overlay to {args.overlay}")
        except (OSError, ValueError, PipelineFailure) as error:
            failures += 1
            print(f"pipeline failed for {image}: {error}", file=sys.stderr)
            if not args.continue_on_error:
                return 2
    return 2 if failures else 0
