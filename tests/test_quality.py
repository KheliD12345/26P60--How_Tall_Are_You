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
    assert assessment.metrics.pose_severity == 0.0


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


def test_blur_score_accepts_grayscale_images_and_clamps_sharpness():
    image = np.zeros((80, 80), dtype=np.uint8)
    image[::2, :] = 255

    score = AcquisitionQualityGate.blur_score(image)

    assert 0.0 <= score <= 1.0
    assert score == 1.0


def test_marker_visibility_clamps_and_handles_zero_expected_markers():
    gate = AcquisitionQualityGate()

    assert gate.evaluate(
        np.zeros((20, 20), dtype=np.uint8),
        detected_markers=8,
    ).metrics.marker_visibility == 1.0
    assert gate.evaluate(
        np.zeros((20, 20), dtype=np.uint8),
        detected_markers=-1,
    ).metrics.marker_visibility == 0.0
    assert gate.evaluate(
        np.zeros((20, 20), dtype=np.uint8),
        detected_markers=0,
        expected_markers=0,
    ).metrics.marker_visibility == 0.0


def test_upright_head_has_no_tilt_severity():
    keypoints = {
        "head_top": Landmark(0.5, 0.1),
        "head_bottom": Landmark(0.5, 0.2),
    }

    assert AcquisitionQualityGate.pose_severity(keypoints) == 0.0


def test_bent_knee_increases_posture_severity():
    upright = {
        "left_hip": Landmark(0.4, 0.4),
        "left_knee": Landmark(0.4, 0.65),
        "left_ankle": Landmark(0.4, 0.9),
    }
    bent = {
        "left_hip": Landmark(0.4, 0.4),
        "left_knee": Landmark(0.55, 0.65),
        "left_ankle": Landmark(0.4, 0.9),
    }

    assert AcquisitionQualityGate.pose_severity(upright) == 0.0
    assert AcquisitionQualityGate.pose_severity(bent) > 0.0


def test_missing_pose_keypoints_use_conservative_unknown_severity():
    assert AcquisitionQualityGate.pose_severity({}) == 0.5
    assert AcquisitionQualityGate.pose_severity({"left_hip": Landmark(0.5, 0.5)}) == 0.5


def test_segmentation_visibility_is_preferred_over_keypoint_confidence():
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    keypoints = {"left_hip": Landmark(0.5, 0.5, visibility=0.1)}

    assessment = AcquisitionQualityGate().evaluate(
        image,
        detected_markers=4,
        keypoints=keypoints,
        segmentation_mask=mask,
        body_bbox=(2, 2, 8, 8),
    )

    assert assessment.metrics.occlusion_score == 1.0


def test_keypoint_confidence_is_used_when_segmentation_is_unavailable():
    keypoints = {
        "left_hip": Landmark(0.5, 0.5, visibility=0.8),
        "right_hip": {"x": 0.6, "y": 0.5, "confidence": 0.6},
    }

    assessment = AcquisitionQualityGate().evaluate(
        np.zeros((20, 20), dtype=np.uint8),
        detected_markers=4,
        keypoints=keypoints,
    )

    assert assessment.metrics.occlusion_score == 0.7


def test_invalid_segmentation_inputs_return_safe_visibility_scores():
    gate = AcquisitionQualityGate()
    image = np.zeros((10, 10), dtype=np.uint8)

    for mask, bbox in (
        (np.zeros((0, 0), dtype=np.uint8), (0, 0, 1, 1)),
        (np.ones((10, 10), dtype=np.uint8), (8, 8, 2, 2)),
        (np.ones((10, 10), dtype=np.uint8), (-2, 0, 2, 2)),
    ):
        assessment = gate.evaluate(
            image,
            detected_markers=4,
            segmentation_mask=mask,
            body_bbox=bbox,
        )
        assert 0.0 <= assessment.metrics.occlusion_score <= 1.0


def test_quality_recommendations_are_stable_for_failed_criteria():
    assessment = AcquisitionQualityGate().evaluate(
        np.zeros((20, 20), dtype=np.uint8),
        detected_markers=0,
        keypoints={},
    )

    assert assessment.recommendations == (
        "Image is too blurry",
        "Required ArUco markers are not all visible",
        "Body landmarks are partially occluded",
    )