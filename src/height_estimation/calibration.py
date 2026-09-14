from pathlib import Path

from .aruco import detect_markers, validate_markers
from .geometry import (
    calculate_pairwise_geometry,
    estimate_cm_per_pixel,
    estimate_homography,
)
from .models import CalibrationResult, MarkerLayout
from .person import detect_person_endpoints


def calibrate_image(
    image_path: str | Path,
    layout: MarkerLayout,
) -> CalibrationResult:
    markers = detect_markers(image_path, layout)
    validate_markers(markers, layout)
    geometry = calculate_pairwise_geometry(markers, layout)
    cm_per_pixel = estimate_cm_per_pixel(geometry)
    homography = estimate_homography(markers, layout)
    person = detect_person_endpoints(image_path)
    return CalibrationResult(
        markers=markers,
        geometry=geometry,
        cm_per_pixel=cm_per_pixel,
        homography=homography,
        person=person,
    )