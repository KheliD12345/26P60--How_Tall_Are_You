from pathlib import Path

import cv2
import numpy as np

from .models import CalibrationResult


def write_calibration_overlay(
    image_path: str | Path,
    calibration: CalibrationResult,
    output_path: str | Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")

    marker_by_id = {marker.id: marker for marker in calibration.markers}
    overlay = image.copy()
    for marker in calibration.markers:
        points = np.array(marker.corners, dtype=np.int32)
        cv2.polylines(overlay, [points], True, (0, 180, 0), 3)
        center = (round(marker.center_x), round(marker.center_y))
        cv2.circle(overlay, center, 7, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            str(marker.id),
            (center[0] + 10, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    for pair in calibration.geometry:
        first = marker_by_id[pair.first_id]
        second = marker_by_id[pair.second_id]
        start = (round(first.center_x), round(first.center_y))
        end = (round(second.center_x), round(second.center_y))
        cv2.line(overlay, start, end, (255, 180, 0), 1)

    cv2.putText(
        overlay,
        f"scale: {calibration.cm_per_pixel:.6f} cm/pixel",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"reprojection: {calibration.homography.reprojection_error_cm:.4f} cm",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise ValueError(f"could not write overlay: {output_path}")