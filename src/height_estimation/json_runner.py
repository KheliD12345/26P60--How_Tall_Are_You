from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import excel_runner
from .models import MarkerLayout


DEFAULT_INPUT_DIR = excel_runner.DEFAULT_INPUT_DIR
DEFAULT_OUTPUT_DIR = excel_runner.DEFAULT_OUTPUT_DIR


def run_batch(input_dir: Path, output_dir: Path) -> int:
    selections = excel_runner.discover_images(input_dir.expanduser().resolve())
    layout = MarkerLayout.from_json(excel_runner.DEFAULT_LAYOUT)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    body_detector = excel_runner.build_body_detector()
    failures = 0
    frame_count = 0
    stick_count = 0

    print(
        f"Processing {len(selections)} images with VGGHeads and ViTPose; "
        "HOG fallback disabled."
    )
    for index, selection in enumerate(selections, start=1):
        estimates = excel_runner.estimate_image(
            selection.path,
            body_detector,
            layout,
        )
        primary_height = (
            estimates.frame_cm
            if estimates.frame_cm is not None
            else estimates.stick_cm
        )
        # Keep the legacy field frame-first while reporting both estimates below.
        result = {
            "participant_id": selection.participant_id,
            "image": selection.path.name,
            "geometric_height_cm": primary_height,
            "frame_estimated_height_cm": estimates.frame_cm,
            "stick_estimated_height_cm": estimates.stick_cm,
            "errors": list(estimates.errors),
        }
        output_path = output_dir / f"{selection.participant_id}.json"
        output_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )

        frame_count += int(estimates.frame_cm is not None)
        stick_count += int(estimates.stick_cm is not None)
        if estimates.errors:
            failures += 1
            for error in estimates.errors:
                print(f"{selection.path.name}: {error}", file=sys.stderr)
        frame_text = (
            f"{estimates.frame_cm:.2f} cm"
            if estimates.frame_cm is not None
            else "blank"
        )
        stick_text = (
            f"{estimates.stick_cm:.2f} cm"
            if estimates.stick_cm is not None
            else "blank"
        )
        print(
            f"[{index}/{len(selections)}] {selection.participant_id}: "
            f"frame={frame_text}; stick={stick_text}"
        )

    print(
        f"Saved JSON results to {output_dir}; frame estimates: {frame_count}, "
        f"stick estimates: {stick_count}, failed images: {failures}."
    )
    return 2 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate ArUco frame and stick heights for every input image."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="folder of images (default: inputs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="folder for per-participant JSON results (default: outputs)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_batch(args.input_dir, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
