import cv2
import json
import numpy as np
import pytest

from height_estimation.advanced_models import BodyDetections, Landmark
from height_estimation.body_detection import (
    BodyDetectionResult,
    BodyDetectionOrchestrator,
    DetectionStatus,
    HandDetectorAdapter,
    HeadDetectorAdapter,
    PersonFallbackAdapter,
    PoseDetectorAdapter,
    SegmentationDetectorAdapter,
    UnavailableDetector,
    body_detections_from_person,
    build_body_detection_result,
    merge_body_detections,
    normalise_head_detection,
    normalise_hand_detection,
    normalise_pose_keypoints,
    normalise_segmentation,
)
from height_estimation.models import PersonEndpoints
from height_estimation.quality import AcquisitionQualityGate


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


def test_segmentation_preserves_mask_shape_values_and_hair_endpoints():
    mask = np.zeros((10, 8), dtype=np.uint8)
    mask[2:7, 3:5] = 1

    detections = normalise_segmentation(
        {
            "mask": mask,
            "mask_coordinate_system": "pixel",
            "hair_length": {
                "top": {"y": 0.2},
                "bottom": {"y": 0.6},
            },
        }
    )

    assert np.array_equal(detections.segmentation_mask, mask)
    assert detections.segmentation_mask.shape == (10, 8)
    assert detections.hair_top.y == pytest.approx(0.2)
    assert detections.hair_bottom.y == pytest.approx(0.6)
    assert detections.hair_top.coordinate_system == "normalized"


def test_segmentation_can_extract_endpoint_x_from_mask():
    mask = np.zeros((10, 8), dtype=np.uint8)
    mask[2, 2:6] = 1
    mask[7, 1:5] = 1

    detections = normalise_segmentation(
        {
            "mask": mask,
            "hair_top": 0.2,
            "hair_bottom": 0.7,
        }
    )

    assert detections.hair_top.x == pytest.approx(0.5)
    assert detections.hair_bottom.x == pytest.approx(2.5 / 7.0)


def test_segmentation_adapter_reports_mask_metadata():
    result = SegmentationDetectorAdapter(
        lambda image: {
            "mask": np.ones((12, 9), dtype=np.uint8),
            "mask_coordinate_system": "pixel",
        }
    ).detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result.status == DetectionStatus.SUCCESS
    assert result.metadata["mask_shape"] == (12, 9)
    assert result.metadata["mask_coordinate_system"] == "pixel"


def test_segmentation_rejects_non_numeric_masks():
    with pytest.raises(ValueError, match="numeric"):
        normalise_segmentation({"mask": [["hair"]]})


def test_hand_detection_normalises_subject_relative_wrist_to_middle_tip_pairs():
    detections = normalise_hand_detection(
        {
            "coordinate_system": "normalized",
            "hands": [
                {
                    "handedness": "Left",
                    "confidence": 0.85,
                    "hand_length": {
                        "landmark_0": {"x": 0.2, "y": 0.6},
                        "landmark_12": {"x": 0.25, "y": 0.5},
                    },
                }
            ],
        }
    )

    assert detections.handedness == ("left",)
    assert detections.hand_confidences == (0.85,)
    assert detections.hand_lengths[0][0] == Landmark(0.2, 0.6)
    assert detections.hand_lengths[0][1].to_pixel(200, 400) == (50.0, 200.0)


def test_hand_detection_accepts_all_landmarks_fallback():
    detections = normalise_hand_detection(
        {
            "hands": [
                {
                    "label": "right",
                    "all_landmarks": {
                        "landmark_0": {"x": 10, "y": 20},
                        "landmark_12": {"x": 12, "y": 5},
                    },
                }
            ],
            "normalized": False,
        }
    )

    assert detections.handedness == ("right",)
    assert detections.hand_lengths[0][0].coordinate_system == "pixel"


def test_hand_detection_rejects_image_perspective_labels():
    with pytest.raises(ValueError, match="image-perspective"):
        normalise_hand_detection(
            {
                "handedness_convention": "image",
                "hands": [],
            }
        )


def test_hand_detector_adapter_reports_hand_count():
    result = HandDetectorAdapter(
        lambda image: {
            "hands": [
                {
                    "handedness": "unknown",
                    "hand_length": ((0.1, 0.5), (0.1, 0.4)),
                }
            ]
        }
    ).detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result.status == DetectionStatus.SUCCESS
    assert result.metadata["hand_count"] == 1


def test_orchestrator_merges_stubbed_detectors_and_keeps_optional_statuses():
    orchestrator = BodyDetectionOrchestrator(
        pose_detector=lambda image: {
            "coordinate_system": "normalized",
            "keypoints": {
                "left_hip": (0.4, 0.5, 0.9),
                "left_ankle": (0.4, 0.9, 0.8),
            },
        },
        head_detector=lambda image: {
            "head_bbox": {"x1": 0.3, "y1": 0.1, "x2": 0.5, "y2": 0.2},
            "confidence": 0.9,
        },
        use_person_fallback=False,
    )

    result = orchestrator.detect(np.zeros((40, 30, 3), dtype=np.uint8))

    assert result.status == DetectionStatus.PARTIAL
    assert result.success is True
    assert result.detections.keypoints["left_hip"].visibility == 0.9
    assert result.detections.head_bbox == (0.3, 0.1, 0.5, 0.2)
    assert result.detector_results["segmentation"].status == DetectionStatus.UNAVAILABLE
    assert result.detector_results["hands"].status == DetectionStatus.UNAVAILABLE


def test_orchestrator_accepts_image_paths(tmp_path):
    image_path = tmp_path / "input.png"
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    orchestrator = BodyDetectionOrchestrator(
        pose_detector=lambda image: {
            "keypoints": {"left_hip": (0.5, 0.5)},
        },
        use_person_fallback=False,
    )

    result = orchestrator.detect(image_path)

    assert result.success is True
    assert result.detections.keypoints["left_hip"] == Landmark(0.5, 0.5)


def test_orchestrator_reports_unreadable_and_empty_inputs(tmp_path):
    orchestrator = BodyDetectionOrchestrator(use_person_fallback=False)

    unreadable = orchestrator.detect(tmp_path / "missing.png")
    empty = orchestrator.detect(np.empty((0, 0, 3), dtype=np.uint8))

    assert unreadable.status == DetectionStatus.FAILED
    assert unreadable.detector_results["input"].status == DetectionStatus.FAILED
    assert empty.status == DetectionStatus.FAILED


def test_orchestrator_keeps_no_person_distinct_from_unavailable_detectors():
    orchestrator = BodyDetectionOrchestrator(
        pose_detector=lambda image: {},
        use_person_fallback=False,
    )

    result = orchestrator.detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result.detector_results["pose"].status == DetectionStatus.NO_PERSON
    assert result.detector_results["head"].status == DetectionStatus.UNAVAILABLE
    assert result.status == DetectionStatus.UNAVAILABLE


def test_orchestrator_reports_no_person_when_all_configured_detectors_find_none():
    no_detection = lambda image: {}
    orchestrator = BodyDetectionOrchestrator(
        pose_detector=no_detection,
        head_detector=no_detection,
        segmentation_detector=no_detection,
        hand_detector=no_detection,
        use_person_fallback=False,
    )

    result = orchestrator.detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result.status == DetectionStatus.NO_PERSON
    assert result.success is False


def test_merged_detections_feed_acquisition_quality_gate():
    orchestrator = BodyDetectionOrchestrator(
        pose_detector=lambda image: {
            "keypoints": {
                "left_hip": (0.4, 0.4, 0.9),
                "left_knee": (0.4, 0.6, 0.9),
                "left_ankle": (0.4, 0.9, 0.9),
            }
        },
        segmentation_detector=lambda image: {
            "mask": np.ones((20, 20), dtype=np.uint8),
        },
        use_person_fallback=False,
    )

    result = orchestrator.detect(np.zeros((20, 20, 3), dtype=np.uint8))
    assessment = AcquisitionQualityGate().evaluate(
        np.zeros((20, 20, 3), dtype=np.uint8),
        detected_markers=4,
        keypoints=result.detections.keypoints,
        segmentation_mask=result.detections.segmentation_mask,
        body_bbox=(0, 0, 20, 20),
    )

    assert result.success is True
    assert 0.0 <= assessment.overall_score <= 1.0


def test_orchestrator_isolates_detector_failures():
    def broken_detector(image):
        raise RuntimeError("stub failed")

    orchestrator = BodyDetectionOrchestrator(
        pose_detector=broken_detector,
        head_detector=lambda image: {
            "head_bbox_pixels": (1, 2, 8, 12),
        },
        use_person_fallback=False,
    )

    result = orchestrator.detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result.status == DetectionStatus.PARTIAL
    assert result.detector_results["pose"].status == DetectionStatus.FAILED
    assert "stub failed" in result.detector_results["pose"].diagnostics[0]
    assert result.detections.head_bbox == (1.0, 2.0, 8.0, 12.0)


def test_orchestrator_preserves_segmentation_and_hand_results():
    mask = np.ones((10, 8), dtype=np.uint8)
    orchestrator = BodyDetectionOrchestrator(
        segmentation_detector=lambda image: {
            "mask": mask,
            "hair_top": 0.1,
            "hair_bottom": 0.8,
        },
        hand_detector=lambda image: {
            "hands": [
                {
                    "handedness": "right",
                    "confidence": 0.7,
                    "hand_length": ((0.2, 0.7), (0.2, 0.6)),
                }
            ]
        },
        use_person_fallback=False,
    )

    result = orchestrator.detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert result.status == DetectionStatus.PARTIAL
    assert np.array_equal(result.detections.segmentation_mask, mask)
    assert result.detections.hair_bottom.y == pytest.approx(0.8)
    assert result.detections.handedness == ("right",)
    assert result.detections.hand_confidences == (0.7,)


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


def test_merge_keeps_hand_metadata_aligned_with_selected_pairs():
    primary = BodyDetections(hand_lengths=(((0.1, 0.5), (0.1, 0.4)),))
    secondary = BodyDetections(
        hand_lengths=(((0.2, 0.5), (0.2, 0.4)),),
        handedness=("left",),
        hand_confidences=(0.8,),
    )

    merged = merge_body_detections(primary, secondary)

    assert merged.hand_lengths == primary.hand_lengths
    assert merged.handedness == ()
    assert merged.hand_confidences == ()


def test_merge_rejects_coordinate_system_mismatch():
    with pytest.raises(ValueError, match="coordinate system mismatch"):
        merge_body_detections(
            BodyDetections(keypoints={"left_hip": Landmark(10, 20, normalized=False)}),
            BodyDetections(keypoints={"left_hip": Landmark(0.1, 0.2)}),
        )


def test_merge_rejects_mismatched_head_endpoint_coordinate_systems():
    with pytest.raises(ValueError, match="head endpoints"):
        merge_body_detections(
            BodyDetections(
                head_top=Landmark(10, 20, normalized=False),
            ),
            BodyDetections(
                head_bottom=Landmark(0.1, 0.2),
            ),
        )


def test_body_detection_result_serialises_all_detection_fields():
    result = BodyDetectionResult(
        detections=BodyDetections(
            head_top=Landmark(10, 20, normalized=False),
            head_bottom=Landmark(10, 50, normalized=False),
            head_bbox=(5, 10, 15, 55),
            head_confidence=0.9,
            hair_top=Landmark(10, 5, normalized=False),
            hair_bottom=Landmark(10, 60, normalized=False),
            hand_lengths=(
                (
                    Landmark(0.1, 0.5),
                    Landmark(0.1, 0.4),
                ),
            ),
            handedness=("left",),
            hand_confidences=(0.8,),
            segmentation_mask=np.ones((2, 2), dtype=np.uint8),
        ),
    )

    payload = result.to_dict()

    assert payload["detections"]["head_bbox"] == [5, 10, 15, 55]
    assert payload["detections"]["handedness"] == ["left"]
    assert payload["detections"]["segmentation_mask"] == [[1, 1], [1, 1]]
    assert payload["detections"]["hair_top"]["coordinate_system"] == "pixel"
    json.dumps(payload)


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