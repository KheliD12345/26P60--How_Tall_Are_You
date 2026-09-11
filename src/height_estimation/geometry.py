from math import hypot
from statistics import median

import cv2
import numpy as np

from .models import (
    DetectedMarker,
    HomographyResult,
    MarkerLayout,
    MarkerPairGeometry,
)


def calculate_pairwise_geometry(
    markers: tuple[DetectedMarker, ...],
    layout: MarkerLayout,
) -> tuple[MarkerPairGeometry, ...]:
    detected_by_id = {marker.id: marker for marker in markers}
    missing_ids = [
        marker.id for marker in layout.markers if marker.id not in detected_by_id
    ]
    if missing_ids:
        ids = ", ".join(str(marker_id) for marker_id in missing_ids)
        raise ValueError(f"missing marker detections: {ids}")

    measurements = []
    for index, first in enumerate(layout.markers):
        first_marker = detected_by_id[first.id]
        for second in layout.markers[index + 1 :]:
            second_marker = detected_by_id[second.id]
            pixel_delta = (
                second_marker.center_x - first_marker.center_x,
                second_marker.center_y - first_marker.center_y,
            )
            physical_delta = (
                second.x_cm - first.x_cm,
                second.y_cm - first.y_cm,
            )
            measurements.append(
                MarkerPairGeometry(
                    first_id=first.id,
                    second_id=second.id,
                    pixel_delta=pixel_delta,
                    physical_delta_cm=physical_delta,
                    pixel_distance=hypot(*pixel_delta),
                    physical_distance_cm=hypot(*physical_delta),
                )
            )

    return tuple(measurements)


def estimate_cm_per_pixel(geometry: tuple[MarkerPairGeometry, ...]) -> float:
    ratios = [
        pair.physical_distance_cm / pair.pixel_distance
        for pair in geometry
        if pair.physical_distance_cm > 0 and pair.pixel_distance > 0
    ]
    if not ratios:
        raise ValueError("could not estimate scale from marker geometry")
    return float(median(ratios))


def estimate_homography(
    markers: tuple[DetectedMarker, ...],
    layout: MarkerLayout,
) -> HomographyResult:
    if len(layout.markers) < 4:
        raise ValueError("at least four marker positions are required")

    detected_by_id = {marker.id: marker for marker in markers}
    missing_ids = [
        marker.id for marker in layout.markers if marker.id not in detected_by_id
    ]
    if missing_ids:
        ids = ", ".join(str(marker_id) for marker_id in missing_ids)
        raise ValueError(f"missing marker detections: {ids}")

    pixel_points = np.array(
        [
            [detected_by_id[marker.id].center_x, detected_by_id[marker.id].center_y]
            for marker in layout.markers
        ],
        dtype=np.float32,
    )
    physical_points = np.array(
        [[marker.x_cm, marker.y_cm] for marker in layout.markers],
        dtype=np.float32,
    )
    matrix, _ = cv2.findHomography(pixel_points, physical_points, method=0)
    if matrix is None:
        raise ValueError("could not calculate marker homography")

    projected_points = cv2.perspectiveTransform(pixel_points.reshape(-1, 1, 2), matrix)
    projected_points = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(projected_points - physical_points, axis=1)
    return HomographyResult(
        matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        reprojection_error_cm=float(np.mean(errors)),
    )


def transform_point(
    point: tuple[float, float],
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[float, float]:
    source = np.array([[point]], dtype=np.float32)
    homography = np.array(matrix, dtype=np.float32)
    transformed = cv2.perspectiveTransform(source, homography)[0, 0]
    return float(transformed[0]), float(transformed[1])