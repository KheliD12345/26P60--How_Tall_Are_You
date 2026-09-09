import argparse
from pathlib import Path
from typing import Sequence

from . import __version__
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = MarkerLayout.from_json(args.layout)
    print(f"Loaded {len(layout.markers)} markers from {args.layout}")
    print(f"Dictionary: {layout.dictionary}")
    print(f"Marker IDs: {', '.join(map(str, layout.marker_ids))}")
    return 0
