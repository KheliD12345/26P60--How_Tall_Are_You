import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .aruco import detect_markers, validate_markers
from .models import MarkerLayout


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
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, help="path for detection JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = MarkerLayout.from_json(args.layout)
    if args.image is not None:
        markers = detect_markers(args.image, layout)
        try:
            validate_markers(markers, layout)
        except ValueError as error:
            build_parser().error(str(error))
        result = {
            "image": str(args.image),
            "markers": [marker.to_dict() for marker in markers],
        }
        output = json.dumps(result, indent=2)
        if args.output is None:
            print(output)
        else:
            args.output.write_text(output + "\n", encoding="utf-8")
            print(f"Wrote {len(markers)} markers to {args.output}")
        return 0

    print(f"Loaded {len(layout.markers)} markers from {args.layout}")
    print(f"Dictionary: {layout.dictionary}")
    print(f"Marker IDs: {', '.join(map(str, layout.marker_ids))}")
    return 0
