from pathlib import Path

import cv2
import numpy as np

from .models import DetectedMarker, MarkerLayout

_DETECTION_SCALES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def _get_dictionary(name: str) -> cv2.aruco.Dictionary:
    try:
        dictionary_id = getattr(cv2.aruco, name)
    except AttributeError as error:
        raise ValueError(f"unknown ArUco dictionary: {name}") from error
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def detect_markers(
    image_path: str | Path,
    layout: MarkerLayout,
) -> tuple[DetectedMarker, ...]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")

    return detect_markers_in_image(image, layout)


def detect_markers_in_image(
    image: np.ndarray,
    layout: MarkerLayout,
) -> tuple[DetectedMarker, ...]:
    return detect_markers_for_dictionary(
        image,
        layout.dictionary,
        required_ids=layout.marker_ids,
    )


def detect_markers_for_dictionary(
    image: np.ndarray,
    dictionary_name: str,
    *,
    required_ids: tuple[int, ...] = (),
) -> tuple[DetectedMarker, ...]:
    if image is None or image.size == 0:
        raise ValueError("image must contain pixel data")

    detector = cv2.aruco.ArucoDetector(
        _get_dictionary(dictionary_name),
        _detector_parameters(),
    )
    detected_by_id: dict[int, tuple[float, DetectedMarker]] = {}
    required = set(required_ids)
    for scale in _DETECTION_SCALES:
        interpolations = (
            (None,)
            if scale == 1.0
            else (cv2.INTER_CUBIC, cv2.INTER_LANCZOS4)
        )
        for interpolation in interpolations:
            scaled_image = (
                image
                if interpolation is None
                else cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=interpolation,
                )
            )
            corners, ids, _ = detector.detectMarkers(scaled_image)
            if ids is None:
                continue

            detected_ids = [int(marker_id) for marker_id in ids.flatten()]
            if len(set(detected_ids)) != len(detected_ids):
                raise ValueError("duplicate ArUco marker IDs were detected")

            for marker_corners, marker_id in zip(corners, detected_ids):
                points = (
                    np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
                    / scale
                )
                center_x, center_y = points.mean(axis=0)
                detected_by_id[marker_id] = (
                    scale,
                    DetectedMarker(
                        id=marker_id,
                        corners=tuple((float(x), float(y)) for x, y in points),
                        center_x=float(center_x),
                        center_y=float(center_y),
                        area_px=float(abs(cv2.contourArea(points.astype(np.float32)))),
                    ),
                )
            if required and required.issubset(detected_by_id):
                break
        if required and required.issubset(detected_by_id):
            break

    return tuple(
        detected_by_id[marker_id][1]
        for marker_id in sorted(detected_by_id)
    )


def _detector_parameters() -> cv2.aruco.DetectorParameters:
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMax = 51
    return parameters


def validate_markers(
    markers: tuple[DetectedMarker, ...],
    layout: MarkerLayout,
) -> None:
    layout.validate()
    detected_ids = [marker.id for marker in markers]
    if len(set(detected_ids)) != len(detected_ids):
        raise ValueError("duplicate ArUco marker IDs were detected")

    missing_ids = sorted(set(layout.marker_ids) - set(detected_ids))
    if missing_ids:
        ids = ", ".join(str(marker_id) for marker_id in missing_ids)
        raise ValueError(f"missing required ArUco markers: {ids}")