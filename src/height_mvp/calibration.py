from pathlib import Path

from .aruco import detect_markers, validate_markers
from .geometry import (
    calculate_pairwise_geometry,
    estimate_cm_per_pixel,
    estimate_homography,
)
from .models import CalibrationResult, MarkerLayout


def calibrate_image(
    image_path: str | Path,
    layout: MarkerLayout,
) -> CalibrationResult:
    markers = detect_markers(image_path, layout)
    validate_markers(markers, layout)
    geometry = calculate_pairwise_geometry(markers, layout)
    cm_per_pixel = estimate_cm_per_pixel(geometry)
    homography = estimate_homography(markers, layout)
    return CalibrationResult(
        markers=markers,
        geometry=geometry,
        cm_per_pixel=cm_per_pixel,
        homography=homography,
    )