import cv2
import numpy as np

from .models import CameraCalibration


def undistort_image(
    image: np.ndarray,
    calibration: CameraCalibration | None = None,
) -> np.ndarray:
    if calibration is None:
        return image

    camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float32)
    distortion_coefficients = np.asarray(
        calibration.distortion_coefficients,
        dtype=np.float32,
    )
    height, width = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        (width, height),
        1,
        (width, height),
    )
    return cv2.undistort(
        image,
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix,
    )