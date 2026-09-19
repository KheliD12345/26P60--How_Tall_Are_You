from math import hypot
from statistics import median

import cv2
import numpy as np

from .models import (
    DetectedMarker,
    HomographyResult,
    MarkerLayout,
    MarkerPairGeometry,
    PersonEndpoints,
)


def calculate_pairwise_geometry(
    markers: tuple[DetectedMarker, ...],
    layout: MarkerLayout,
) -> tuple[MarkerPairGeometry, ...]:
    layout.validate()
    detected_by_id = {marker.id: marker for marker in markers}
    if len(detected_by_id) != len(markers):
        raise ValueError("duplicate marker detections cannot define geometry")
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
            pixel_distance = hypot(*pixel_delta)
            physical_distance_cm = hypot(*physical_delta)
            if pixel_distance <= 1e-6:
                raise ValueError("marker pixel distances must be greater than zero")
            if physical_distance_cm <= 1e-6:
                raise ValueError(
                    "marker physical distances must be greater than zero"
                )
            measurements.append(
                MarkerPairGeometry(
                    first_id=first.id,
                    second_id=second.id,
                    pixel_delta=pixel_delta,
                    physical_delta_cm=physical_delta,
                    pixel_distance=pixel_distance,
                    physical_distance_cm=physical_distance_cm,
                )
            )

    return tuple(measurements)


def _marker_size_cm_per_pixel(
    markers: tuple[DetectedMarker, ...],
    marker_size_cm: float,
) -> float:
    if not np.isfinite(marker_size_cm) or marker_size_cm <= 0:
        raise ValueError("marker size must be finite and greater than zero")

    marker_sizes = []
    for marker in markers:
        corners = np.asarray(marker.corners, dtype=np.float64)
        side_lengths = np.linalg.norm(
            np.roll(corners, -1, axis=0) - corners,
            axis=1,
        )
        if not np.isfinite(side_lengths).all() or np.any(side_lengths <= 0):
            raise ValueError("marker side lengths must be finite and greater than zero")
        marker_sizes.append(float(np.mean(side_lengths)))

    if not marker_sizes:
        raise ValueError("at least one marker is required for marker-size scale")
    return float(marker_size_cm / np.median(marker_sizes))


def estimate_cm_per_pixel(
    geometry: tuple[MarkerPairGeometry, ...],
    *,
    markers: tuple[DetectedMarker, ...] | None = None,
    marker_size_cm: float | None = None,
) -> float:
    ratios = [
        pair.physical_distance_cm / pair.pixel_distance
        for pair in geometry
        if pair.physical_distance_cm > 1e-6 and pair.pixel_distance > 1e-6
    ]
    if not ratios or not all(np.isfinite(ratio) and ratio > 0 for ratio in ratios):
        raise ValueError("could not estimate scale from marker geometry")
    geometry_scale = float(median(ratios))
    if markers is None and marker_size_cm is None:
        return geometry_scale
    if markers is None or marker_size_cm is None:
        raise ValueError("markers and marker_size_cm must be provided together")
    marker_scale = _marker_size_cm_per_pixel(markers, marker_size_cm)
    return float(np.mean((geometry_scale, marker_scale)))


def estimate_homography(
    markers: tuple[DetectedMarker, ...],
    layout: MarkerLayout,
) -> HomographyResult:
    if len(layout.markers) < 4:
        raise ValueError("at least four marker positions are required")
    layout.validate()

    detected_by_id = {marker.id: marker for marker in markers}
    if len(detected_by_id) != len(markers):
        raise ValueError("duplicate marker detections cannot define homography")
    missing_ids = [
        marker.id for marker in layout.markers if marker.id not in detected_by_id
    ]
    if missing_ids:
        ids = ", ".join(str(marker_id) for marker_id in missing_ids)
        raise ValueError(f"missing marker detections: {ids}")

    pixel_points = np.array(
        [
            [
                detected_by_id[marker.id].center_x,
                detected_by_id[marker.id].center_y,
            ]
            for marker in layout.markers
        ],
        dtype=np.float32,
    )
    physical_points = np.array(
        [[marker.x_cm, marker.y_cm] for marker in layout.markers],
        dtype=np.float32,
    )
    if not np.isfinite(pixel_points).all():
        raise ValueError("marker pixel coordinates must be finite")
    if np.linalg.matrix_rank(pixel_points - pixel_points[0]) < 2:
        raise ValueError("marker pixel coordinates are collinear")
    if np.linalg.matrix_rank(physical_points - physical_points[0]) < 2:
        raise ValueError("marker physical positions are collinear")

    try:
        matrix, _ = cv2.findHomography(pixel_points, physical_points, method=0)
    except cv2.error as error:
        raise ValueError("could not calculate marker homography") from error
    if matrix is None:
        raise ValueError("could not calculate marker homography")
    if not np.isfinite(matrix).all():
        raise ValueError("marker homography must contain finite values")

    projected_points = cv2.perspectiveTransform(
        pixel_points.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    if not np.isfinite(projected_points).all():
        raise ValueError("marker homography produced non-finite coordinates")
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


def transform_person_endpoints(
    person: PersonEndpoints,
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        transform_point(person.top_of_head, matrix),
        transform_point(person.bottom_of_feet, matrix),
    )