import numpy as np
import pytest

from height_estimation.advanced_models import BodyDetections, Landmark
from height_estimation.body_detection import (
    BodyDetectionResult,
    DetectionStatus,
    HeadDetectorAdapter,
    PersonFallbackAdapter,
    PoseDetectorAdapter,
    UnavailableDetector,
    body_detections_from_person,
    build_body_detection_result,
    merge_body_detections,
    normalise_head_detection,
    normalise_pose_keypoints,
)
from height_estimation.models import PersonEndpoints


def test_pose_keypoints_preserve_normalised_coordinates_and_visibility():
    detections = normalise_pose_keypoints(
        {
            "coordinate_system": "normalized",
            "keypoints": {
                "left_hip": {"x": 0.25, "y": 0.4, "confidence": 0.8},
                "right_heel": {"x": 0.7, "y": 0.95, "visibility": 0.9},
            },
        }
    )

    assert detections.keypoints["left_hip"] == Landmark(0.25, 0.4, 0.8)
    assert detections.keypoints["right_heel"].coordinate_system == "normalized"
    assert detections.keypoints["right_heel"].to_pixel(200, 400) == (140.0, 380.0)


def test_pose_keypoints_preserve_pixel_coordinates_and_heel_aliases():
    detections = normalise_pose_keypoints(
        {
            "coordinate_system": "pixel",
            "keypoints": {
                "left_ankle": (20.0, 180.0, 0.7),
                "heel_landmarks": {"right": {"x": 60.0, "y": 181.0}},
            },
        }
    )

    assert detections.keypoints["left_ankle"].coordinate_system == "pixel"
    assert detections.keypoints["left_ankle"].to_pixel(200, 400) == (20.0, 180.0)
    assert detections.heel_landmarks() == (
        detections.keypoints["left_ankle"],
        detections.keypoints["right_heel"],
    )


def test_per_landmark_coordinate_metadata_overrides_wrapper_default():
    detections = normalise_pose_keypoints(
        {
            "coordinate_system": "normalized",
            "keypoints": {
                "left_knee": {"x": 20, "y": 30, "normalized": False},
            },
        }
    )

    assert detections.keypoints["left_knee"].coordinate_system == "pixel"
    assert detections.keypoints["left_knee"].to_pixel(200, 400) == (20.0, 30.0)


def test_head_detection_preserves_pixel_bbox_and_endpoints():
    detections = normalise_head_detection(
        {
            "head_bbox_pixels": {"x1": 40, "y1": 20, "x2": 80, "y2": 70},
            "head_top_y_pixels": 18,
            "confidence": 0.91,
        }
    )

    assert detections.head_bbox == (40.0, 20.0, 80.0, 70.0)
    assert detections.head_bbox_coordinate_system == "pixel"
    assert detections.head_top == Landmark(60.0, 18.0, normalized=False)
    assert detections.head_bottom == Landmark(60.0, 70.0, normalized=False)
    assert detections.head_confidence == 0.91


def test_head_detection_preserves_normalised_bbox_coordinates():
    detections = normalise_head_detection(
        {
            "head_bbox": {"x1": 0.2, "y1": 0.1, "x2": 0.4, "y2": 0.3},
            "head_top_y": 0.08,
        }
    )

    assert detections.head_bbox_coordinate_system == "normalized"
    assert detections.head_top.x == pytest.approx(0.3)
    assert detections.head_top.y == pytest.approx(0.08)
    assert detections.head_bottom.x == pytest.approx(0.3)
    assert detections.head_bottom.y == pytest.approx(0.3)


def test_person_fallback_keeps_endpoints_and_bounds_confidence():
    person = PersonEndpoints(
        box=(10, 20, 80, 180),
        top_of_head=(50.0, 20.0),
        bottom_of_feet=(50.0, 199.0),
        score=1.5,
    )

    detections = body_detections_from_person(person)

    assert detections.head_top.coordinate_system == "pixel"
    assert detections.heel_landmarks()[0].y == 199.0
    assert detections.heel_landmarks()[0].visibility == 1.0


def test_optional_detector_is_explicitly_unavailable():
    result = UnavailableDetector("head", "VGGHeads is not installed").detect(
        np.zeros((20, 20, 3), dtype=np.uint8)
    )

    assert result.status == DetectionStatus.UNAVAILABLE
    assert result.available is False
    assert "not installed" in result.diagnostics[0]


def test_adapters_report_invalid_output_as_failure():
    result = PoseDetectorAdapter(
        lambda image: {"keypoints": {"left_hip": {"x": 0.5}}}
    ).detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result.status == DetectionStatus.FAILED
    assert "invalid pose output" in result.diagnostics[0]


def test_adapters_report_empty_output_as_no_person():
    result = HeadDetectorAdapter(lambda image: {}).detect(
        np.zeros((20, 20, 3), dtype=np.uint8)
    )

    assert result.status == DetectionStatus.NO_PERSON


def test_partial_result_retains_valid_detections_and_diagnostics():
    pose = PoseDetectorAdapter(
        lambda image: {
            "keypoints": {"left_hip": (0.4, 0.5, 0.9)},
        }
    ).detect(np.zeros((20, 20, 3), dtype=np.uint8))
    unavailable = UnavailableDetector("head").detect(
        np.zeros((20, 20, 3), dtype=np.uint8)
    )

    result = build_body_detection_result(
        {"pose": pose, "head": unavailable}
    )

    assert isinstance(result, BodyDetectionResult)
    assert result.status == DetectionStatus.PARTIAL
    assert result.success is True
    assert result.detections.keypoints["left_hip"].visibility == 0.9
    assert result.diagnostics
    assert result.detector_results["pose"].metadata["total_keypoints"] == 1
    assert result.to_dict()["detectors"]["head"]["status"] == "unavailable"


def test_merge_does_not_replace_valid_values_with_empty_fallback():
    valid = BodyDetections(
        keypoints={"left_hip": Landmark(0.4, 0.5)},
        head_top=Landmark(0.4, 0.1),
    )

    merged = merge_body_detections(valid, BodyDetections())

    assert merged.keypoints["left_hip"] == valid.keypoints["left_hip"]
    assert merged.head_top == valid.head_top


def test_merge_rejects_coordinate_system_mismatch():
    with pytest.raises(ValueError, match="coordinate system mismatch"):
        merge_body_detections(
            BodyDetections(keypoints={"left_hip": Landmark(10, 20, normalized=False)}),
            BodyDetections(keypoints={"left_hip": Landmark(0.1, 0.2)}),
        )


def test_person_fallback_adapter_accepts_numpy_images(monkeypatch):
    person = PersonEndpoints(
        box=(1, 2, 10, 30),
        top_of_head=(6.0, 2.0),
        bottom_of_feet=(6.0, 31.0),
        score=0.8,
    )
    monkeypatch.setattr(
        "height_estimation.body_detection.detect_person_endpoints",
        lambda image: person,
    )

    result = PersonFallbackAdapter().detect(
        np.zeros((40, 20, 3), dtype=np.uint8)
    )

    assert result.status == DetectionStatus.SUCCESS
    assert result.detections.head_top == Landmark(
        6.0,
        2.0,
        visibility=0.8,
        normalized=False,
    )