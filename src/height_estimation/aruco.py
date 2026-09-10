from pathlib import Path

import cv2
import numpy as np

from .models import DetectedMarker, MarkerLayout


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

    detector = cv2.aruco.ArucoDetector(
        _get_dictionary(layout.dictionary),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return ()

    markers = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        center_x, center_y = points.mean(axis=0)
        markers.append(
            DetectedMarker(
                id=int(marker_id),
                corners=tuple((float(x), float(y)) for x, y in points),
                center_x=float(center_x),
                center_y=float(center_y),
                area_px=float(abs(cv2.contourArea(points))),
            )
        )

    return tuple(sorted(markers, key=lambda marker: marker.id))


def validate_markers(
    markers: tuple[DetectedMarker, ...],
    layout: MarkerLayout,
) -> None:
    detected_ids = [marker.id for marker in markers]
    if len(set(detected_ids)) != len(detected_ids):
        raise ValueError("duplicate ArUco marker IDs were detected")

    missing_ids = sorted(set(layout.marker_ids) - set(detected_ids))
    if missing_ids:
        ids = ", ".join(str(marker_id) for marker_id in missing_ids)
        raise ValueError(f"missing required ArUco markers: {ids}")