import cv2
import numpy as np

from height_estimation.advanced_models import Landmark, QualityLevel
from height_estimation.quality import AcquisitionQualityGate


def test_quality_gate_accepts_sharp_upright_image():
    image = np.random.default_rng(4).integers(
        0, 255, size=(240, 160, 3), dtype=np.uint8
    )
    keypoints = {
        "left_hip": Landmark(0.4, 0.4),
        "left_knee": Landmark(0.4, 0.65),
        "left_ankle": Landmark(0.4, 0.9),
        "head_top": Landmark(0.4, 0.1),
        "head_bottom": Landmark(0.4, 0.2),
    }

    assessment = AcquisitionQualityGate().evaluate(
        image,
        detected_markers=4,
        keypoints=keypoints,
    )

    assert assessment.passed
    assert assessment.metrics.level() == QualityLevel.HIGH


def test_quality_gate_rejects_blurred_image():
    image = np.full((240, 160, 3), 128, dtype=np.uint8)
    image = cv2.GaussianBlur(image, (31, 31), 0)

    assessment = AcquisitionQualityGate().evaluate(
        image,
        detected_markers=2,
        keypoints={},
    )

    assert not assessment.passed
    assert "Image is too blurry" in assessment.recommendations
    assert assessment.metrics.marker_visibility == 0.5