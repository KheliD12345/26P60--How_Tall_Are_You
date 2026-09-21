from pathlib import Path

import cv2

from .aruco import detect_markers, detect_markers_in_image, validate_markers
from .camera import undistort_image
from .geometry import (
    calculate_pairwise_geometry,
    estimate_cm_per_pixel,
    estimate_homography,
    transform_person_endpoints,
)
from .height import (
    calculate_perspective_height_cm,
    validate_perspective_result,
)
from .models import CalibrationResult, CameraCalibration, MarkerLayout
from .person import detect_person_endpoints


def calibrate_image(
    image_path: str | Path,
    layout: MarkerLayout,
    camera_calibration: CameraCalibration | None = None,
    camera_calibration_path: str | Path | None = None,
) -> CalibrationResult:
    if camera_calibration is not None and camera_calibration_path is not None:
        raise ValueError(
            "provide camera_calibration or camera_calibration_path, not both"
        )

    working_image = None
    if camera_calibration is None and camera_calibration_path is None:
        markers = detect_markers(image_path, layout)
    else:
        if camera_calibration_path is not None:
            camera_calibration = CameraCalibration.from_json(camera_calibration_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        working_image = undistort_image(image, camera_calibration)
        markers = detect_markers_in_image(
            working_image,
            layout,
        )
    validate_markers(markers, layout)
    geometry = calculate_pairwise_geometry(markers, layout)
    cm_per_pixel = estimate_cm_per_pixel(
        geometry,
        markers=markers,
        marker_size_cm=layout.marker_size_cm,
    )
    homography = estimate_homography(markers, layout)
    person = detect_person_endpoints(
        working_image if working_image is not None else image_path
    )
    perspective_height_cm = None
    if person is not None:
        head, feet = transform_person_endpoints(person, homography.matrix)
        try:
            validate_perspective_result(
                head,
                feet,
                homography.reprojection_error_cm,
            )
            perspective_height_cm = calculate_perspective_height_cm(head, feet)
        except ValueError:
            perspective_height_cm = None

    return CalibrationResult(
        markers=markers,
        geometry=geometry,
        cm_per_pixel=cm_per_pixel,
        homography=homography,
        person=person,
        perspective_height_cm=perspective_height_cm,
        camera_calibration=camera_calibration,
        diagnostics=(
            "Camera undistortion applied before marker detection."
            if camera_calibration is not None
            else "Camera calibration was not supplied."
        ,),
        working_image=working_image,
    )


def save_calibration(result: CalibrationResult, path: str | Path) -> None:
    result.to_json(path)


def load_calibration(path: str | Path) -> CalibrationResult:
    return CalibrationResult.from_json(path)