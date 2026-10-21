from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Sequence
from zipfile import BadZipFile

import cv2
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import MergedCell
from openpyxl.worksheet.worksheet import Worksheet

from .body_detection import (
    BodyDetectionConfig,
    BodyDetectionOrchestrator,
    VGGHeadsDetectorAdapter,
    ViTPoseDetectorAdapter,
)
from .frame_stick import (
    FrameStickEstimate,
    SubjectEndpoints,
    detect_markers,
    detect_subject,
    estimate_height,
    require_markers,
)
from .models import MarkerLayout


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_LAYOUT = PROJECT_ROOT / "configs" / "marker_layout.json"
DEFAULT_WORKBOOK = DEFAULT_OUTPUT_DIR / "Participant Heights.xlsx"
STICK_MARKER_IDS = (4, 5)
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
PARTICIPANT_PREFIX = re.compile(r"^P(?P<number>\d+)(?:$|[^A-Za-z0-9])", re.IGNORECASE)
PARTICIPANT_ID = re.compile(r"^P(?P<number>\d+)$", re.IGNORECASE)
IMAGE_ERRORS = (OSError, ValueError, RuntimeError, ImportError, cv2.error)
WORKBOOK_HEADERS = (
    "Participant Number",
    "Measured Height (cm)",
    "Frame Estimated Height (cm)",
    "Stick Estimated Height (cm)",
)
HEADER_ROLES = {
    "participant number": "participant",
    "measured height (cm)": "measured",
    "frame estimated height (cm)": "frame",
    "stick estimated height (cm)": "stick",
}


@dataclass(frozen=True)
class ParticipantImage:
    participant_id: str
    path: Path


@dataclass(frozen=True)
class ImageEstimates:
    frame_cm: float | None
    stick_cm: float | None
    errors: tuple[str, ...] = ()


def normalise_participant_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = PARTICIPANT_ID.fullmatch(value.strip())
    if match is None:
        return None
    return f"P{int(match.group('number')):03d}"


def discover_images(input_dir: Path) -> list[ParticipantImage]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {input_dir}")

    paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not paths:
        raise ValueError(f"no supported images found in {input_dir}")

    labelled: dict[Path, str] = {}
    used_ids: dict[str, Path] = {}
    for path in paths:
        match = PARTICIPANT_PREFIX.match(path.stem)
        if match is None:
            continue
        participant_id = f"P{int(match.group('number')):03d}"
        previous_path = used_ids.get(participant_id)
        if previous_path is not None:
            raise ValueError(
                f"duplicate participant ID {participant_id}: "
                f"{previous_path.name} and {path.name}"
            )
        labelled[path] = participant_id
        used_ids[participant_id] = path

    # Reserve labelled IDs before assigning fallback IDs.
    selections: list[ParticipantImage] = []
    next_number = 1
    for path in paths:
        participant_id = labelled.get(path)
        if participant_id is None:
            participant_id = f"P{next_number:03d}"
            while participant_id in used_ids:
                next_number += 1
                participant_id = f"P{next_number:03d}"
            used_ids[participant_id] = path
            next_number += 1
        selections.append(ParticipantImage(participant_id, path))
    return selections


def build_body_detector() -> BodyDetectionOrchestrator:
    return BodyDetectionOrchestrator(
        configuration=BodyDetectionConfig(
            pose_detector=ViTPoseDetectorAdapter(),
            head_detector=VGGHeadsDetectorAdapter(),
        ),
        use_person_fallback=False,
    )


def _normalise_header(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.casefold().split())


def _find_headers(sheet: Worksheet) -> tuple[int, dict[str, int]]:
    columns: dict[str, int] = {}
    for row_number in range(1, min(sheet.max_row, 20) + 1):
        columns = {}
        for column_number in range(1, sheet.max_column + 1):
            header = _normalise_header(sheet.cell(row_number, column_number).value)
            role = HEADER_ROLES.get(header)
            if role is None:
                continue
            if role in columns:
                raise ValueError(f"duplicate spreadsheet header: {header}")
            columns[role] = column_number
        if len(columns) == len(HEADER_ROLES):
            return row_number, columns

    missing = sorted(set(HEADER_ROLES.values()) - set(columns))
    raise ValueError(
        "could not find the required participant and height columns "
        f"in the first 20 rows; missing: {', '.join(missing)}"
    )


def _participant_rows(
    sheet: Worksheet,
    header_row: int,
    participant_column: int,
) -> dict[str, int]:
    rows: dict[str, int] = {}
    for row_number in range(header_row + 1, sheet.max_row + 1):
        value = sheet.cell(row_number, participant_column).value
        if value is None:
            continue
        participant_id = normalise_participant_id(value)
        if participant_id is None:
            continue
        if participant_id in rows:
            raise ValueError(f"duplicate participant row: {participant_id}")
        rows[participant_id] = row_number
    return rows


def _create_workbook(
    selections: list[ParticipantImage],
) -> tuple[Workbook, Worksheet, dict[str, int], dict[str, int]]:
    workbook = Workbook()
    sheet = workbook.active
    if not isinstance(sheet, Worksheet):
        raise ValueError("new workbook has no active worksheet")
    sheet.title = "Participant Heights"
    sheet.append(WORKBOOK_HEADERS)
    columns = {
        HEADER_ROLES[_normalise_header(header)]: column_number
        for column_number, header in enumerate(WORKBOOK_HEADERS, start=1)
    }
    rows: dict[str, int] = {}
    for selection in selections:
        sheet.append([selection.participant_id, None, None, None])
        rows[selection.participant_id] = sheet.max_row
    return workbook, sheet, columns, rows


def _set_cell_value(
    sheet: Worksheet,
    row_number: int,
    column_number: int,
    value: str | float | None,
) -> None:
    cell = sheet.cell(row_number, column_number)
    if isinstance(cell, MergedCell):
        raise ValueError(f"cannot write to merged cell {cell.coordinate}")
    cell.value = value


def _height_cm(estimate: FrameStickEstimate) -> float:
    value = estimate["height_cm"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("height estimator returned a non-numeric result")
    height = float(value)
    if not isfinite(height) or height <= 0:
        raise ValueError("height estimator returned an invalid result")
    return round(height, 2)


def _estimate_frame(
    image: np.ndarray,
    subject: SubjectEndpoints,
    layout: MarkerLayout,
) -> float:
    detections = detect_markers(image, layout.dictionary, layout.marker_ids)
    require_markers(detections, layout.marker_ids, "frame")
    return _height_cm(estimate_height(detections, None, subject, layout))


def _estimate_stick(
    image: np.ndarray,
    subject: SubjectEndpoints,
    layout: MarkerLayout,
) -> float:
    detections = detect_markers(image, layout.dictionary, STICK_MARKER_IDS)
    first, second = require_markers(detections, STICK_MARKER_IDS, "stick")
    return _height_cm(
        estimate_height({}, (first, second), subject, layout)
    )


def estimate_image(
    path: Path,
    body_detector: BodyDetectionOrchestrator,
    layout: MarkerLayout,
) -> ImageEstimates:
    try:
        image = cv2.imread(str(path))
    except (OSError, cv2.error) as error:
        return ImageEstimates(None, None, (f"image: {error}",))
    if image is None:
        return ImageEstimates(None, None, (f"image: could not read {path}",))

    try:
        subject, _ = detect_subject(body_detector, image)
    except IMAGE_ERRORS as error:
        return ImageEstimates(None, None, (f"body detection: {error}",))

    errors: list[str] = []
    frame_cm: float | None = None
    stick_cm: float | None = None
    try:
        frame_cm = _estimate_frame(image, subject, layout)
    except IMAGE_ERRORS as error:
        errors.append(f"frame: {error}")
    try:
        stick_cm = _estimate_stick(image, subject, layout)
    except IMAGE_ERRORS as error:
        errors.append(f"stick: {error}")
    return ImageEstimates(frame_cm, stick_cm, tuple(errors))


def _save_workbook(workbook: Workbook, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        workbook.save(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_batch(workbook_path: Path, input_dir: Path) -> int:
    workbook_path = workbook_path.expanduser().resolve()
    input_dir = input_dir.expanduser().resolve()
    selections = discover_images(input_dir)
    if workbook_path.suffix.lower() != ".xlsx":
        raise ValueError(f"workbook must be an .xlsx file: {workbook_path}")
    layout = MarkerLayout.from_json(DEFAULT_LAYOUT)
    if workbook_path.exists():
        if not workbook_path.is_file():
            raise ValueError(f"workbook path is not a file: {workbook_path}")
        workbook = load_workbook(workbook_path)
        sheet = workbook.active
        if not isinstance(sheet, Worksheet):
            workbook.close()
            raise ValueError("workbook's active sheet must be a worksheet")
        header_row, columns = _find_headers(sheet)
        rows = _participant_rows(sheet, header_row, columns["participant"])
    else:
        workbook_path.parent.mkdir(parents=True, exist_ok=True)
        workbook, sheet, columns, rows = _create_workbook(selections)

    for selection in selections:
        if selection.participant_id not in rows:
            row_number = sheet.max_row + 1
            _set_cell_value(
                sheet,
                row_number,
                columns["participant"],
                selection.participant_id,
            )
            rows[selection.participant_id] = row_number

    for row_number in rows.values():
        _set_cell_value(sheet, row_number, columns["frame"], None)
        _set_cell_value(sheet, row_number, columns["stick"], None)

    body_detector = build_body_detector()
    failures: list[str] = []
    estimates_by_participant: dict[str, ImageEstimates] = {}

    print(
        f"Processing {len(selections)} images with VGGHeads and ViTPose; "
        "HOG fallback disabled."
    )
    for index, selection in enumerate(selections, start=1):
        estimate = estimate_image(selection.path, body_detector, layout)
        row_number = rows[selection.participant_id]
        _set_cell_value(sheet, row_number, columns["frame"], estimate.frame_cm)
        _set_cell_value(sheet, row_number, columns["stick"], estimate.stick_cm)
        estimates_by_participant[selection.participant_id] = estimate

        for error in estimate.errors:
            message = f"{selection.path.name}: {error}"
            failures.append(message)
            print(message, file=sys.stderr)
        frame_text = (
            f"{estimate.frame_cm:.2f} cm" if estimate.frame_cm is not None else "blank"
        )
        stick_text = (
            f"{estimate.stick_cm:.2f} cm" if estimate.stick_cm is not None else "blank"
        )
        print(
            f"[{index}/{len(selections)}] {selection.path.name}: "
            f"frame={frame_text}; stick={stick_text}"
        )

    _save_workbook(workbook, workbook_path)
    workbook.close()
    frame_count = sum(
        result.frame_cm is not None for result in estimates_by_participant.values()
    )
    stick_count = sum(
        result.stick_cm is not None for result in estimates_by_participant.values()
    )
    print(
        f"Saved {workbook_path}; frame estimates: {frame_count}, "
        f"stick estimates: {stick_count}, failures: {len(failures)}."
    )
    return 2 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write model-only ArUco frame and stick estimates to the participant "
            "Excel workbook."
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=DEFAULT_WORKBOOK,
        help="workbook path (default: outputs/Participant Heights.xlsx)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="folder of participant images (default: inputs)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_batch(args.workbook, args.input_dir)
    except (BadZipFile, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
