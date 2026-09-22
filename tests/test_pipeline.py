from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import cv2
import json

from height_estimation.advanced_models import (
    BodyDetections,
    HeightEstimate,
    MeasurementMethod,
    MeasurementResult,
    QualityAssessment,
    QualityMetrics,
)
from height_estimation.body_detection import (
    BodyDetectionConfig,
    BodyDetectionResult,
    DetectionStatus,
    DetectorResult,
    MediaPipeHandDetectorAdapter,
    MediaPipeSegmentationAdapter,
    PoseDetectorAdapter,
    VGGHeadsDetectorAdapter,
    ViTPoseDetectorAdapter,
)
from height_estimation.models import CalibrationResult, MarkerLayout, MarkerPosition
from height_estimation.fusion import build_measurement_result
from height_estimation.pipeline import (
    PIPELINE_STAGE_ORDER,
    BodyDetectionStage,
    CalibrationStage,
    EstimationStage,
    FusionStage,
    PipelineConfig,
    PipelineDependencies,
    PipelineFailure,
    PipelineFailureCode,
    PipelineInput,
    PipelineRunResult,
    PipelineStage,
    PipelineStageResult,
    PipelineStatus,
    QualityStage,
    MeasurementPipeline,
    default_pipeline_dependencies,
)


def make_layout() -> MarkerLayout:
    return MarkerLayout(
        dictionary="DICT_4X4_50",
        marker_size_cm=20.0,
        markers=(
            MarkerPosition(0, "bottom_left", 0.0, 0.0),
            MarkerPosition(1, "bottom_right", 100.0, 0.0),
            MarkerPosition(2, "top_left", 0.0, 60.0),
            MarkerPosition(3, "top_right", 100.0, 60.0),
        ),
    )


class FakeCalibration:
    def run(self, pipeline_input):
        raise NotImplementedError


class FakeQuality:
    def run(self, image, *, calibration, body=None):
        raise NotImplementedError


class FakeBodyDetection:
    def run(self, image):
        raise NotImplementedError


class FakeEstimation:
    def run(self, detections, *, calibration, quality):
        raise NotImplementedError


class FakeFusion:
    def run(self, estimates, *, quality):
        raise NotImplementedError


def make_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        calibration=FakeCalibration(),
        quality=FakeQuality(),
        body_detection=FakeBodyDetection(),
        estimation=FakeEstimation(),
        fusion=FakeFusion(),
    )


def make_quality() -> QualityAssessment:
    return QualityAssessment(passed=True, metrics=QualityMetrics())


def make_measurement_result() -> MeasurementResult:
    estimate = HeightEstimate(MeasurementMethod.GEOMETRIC, 170.0, 0.9)
    return MeasurementResult(
        estimated_height_cm=170.0,
        uncertainty_range=(165.0, 175.0),
        quality=make_quality(),
        method_estimates=(estimate,),
        fusion_weights={"geometric": 1.0},
    )


def test_pipeline_input_normalises_paths_and_preserves_calibration_option():
    pipeline_input = PipelineInput(
        image_path="images/person.jpg",
        layout=make_layout(),
        camera_calibration_path="configs/camera.json",
    )

    assert pipeline_input.image_path == Path("images/person.jpg")
    assert pipeline_input.camera_calibration_path == Path("configs/camera.json")
    assert pipeline_input.layout.marker_ids == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"quality_policy": "reject"}, "quality policy"),
        ({"confidence_level": 0.0}, "confidence level"),
        ({"confidence_level": 1.0}, "confidence level"),
        ({"use_person_fallback": "yes"}, "use_person_fallback"),
    ],
)
def test_pipeline_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig(**kwargs)


def test_pipeline_config_rejects_invalid_body_detection_configuration():
    with pytest.raises(TypeError, match="BodyDetectionConfig"):
        PipelineConfig(body_detection=object())


def test_default_pipeline_dependencies_propagate_body_detection_configuration():
    pose = PoseDetectorAdapter(
        lambda image: {"keypoints": {"left_hip": (0.4, 0.5)}}
    )
    configuration = BodyDetectionConfig(pose_detector=pose)
    dependencies = default_pipeline_dependencies(
        PipelineConfig(
            body_detection=configuration,
            use_person_fallback=False,
        )
    )

    result = dependencies.body_detection.run(
        np.zeros((20, 20, 3), dtype=np.uint8)
    )

    assert result.status is DetectionStatus.PARTIAL
    assert result.detections.keypoints["left_hip"].x == 0.4
    assert result.detector_results["head"].status is DetectionStatus.UNAVAILABLE


def test_default_pipeline_dependencies_keep_optional_adapters_unloaded():
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return lambda image: {"keypoints": {"left_hip": (0.4, 0.5)}}

    adapter = ViTPoseDetectorAdapter(factory)
    dependencies = default_pipeline_dependencies(
        PipelineConfig(
            body_detection=BodyDetectionConfig(pose_detector=adapter),
            use_person_fallback=False,
        )
    )

    assert calls == 0
    result = dependencies.body_detection.run(
        np.zeros((20, 20, 3), dtype=np.uint8)
    )

    assert result.success is True
    assert calls == 1


def test_pipeline_config_requires_directory_for_intermediate_outputs(tmp_path):
    with pytest.raises(ValueError, match="intermediate_output_dir"):
        PipelineConfig(write_intermediate_outputs=True)

    config = PipelineConfig(
        output_path=tmp_path / "result.json",
        intermediate_output_dir=tmp_path / "stages",
        write_intermediate_outputs=True,
    )

    assert config.output_path == tmp_path / "result.json"
    assert config.intermediate_output_dir == tmp_path / "stages"


def test_pipeline_stage_order_captures_quality_reassessment_before_estimation():
    assert PIPELINE_STAGE_ORDER == (
        PipelineStage.INPUT,
        PipelineStage.CALIBRATION,
        PipelineStage.QUALITY_ASSESSMENT,
        PipelineStage.BODY_DETECTION,
        PipelineStage.QUALITY_REASSESSMENT,
        PipelineStage.ESTIMATION,
        PipelineStage.FUSION,
        PipelineStage.PERSISTENCE,
    )
    assert PIPELINE_STAGE_ORDER.index(PipelineStage.QUALITY_REASSESSMENT) < PIPELINE_STAGE_ORDER.index(
        PipelineStage.ESTIMATION
    )


def test_pipeline_dependencies_keep_each_stage_injectable():
    dependencies = make_dependencies()

    assert isinstance(dependencies.calibration, FakeCalibration)
    assert isinstance(dependencies.quality, FakeQuality)
    assert isinstance(dependencies.body_detection, FakeBodyDetection)
    assert isinstance(dependencies.estimation, FakeEstimation)
    assert isinstance(dependencies.fusion, FakeFusion)


def test_stage_result_preserves_partial_values_and_diagnostics():
    body_result = BodyDetectionResult(detections=BodyDetections())
    result = PipelineStageResult(
        stage=PipelineStage.BODY_DETECTION,
        status=PipelineStatus.PARTIAL,
        value=body_result,
        diagnostics=("Pose detector unavailable.",),
        warnings=("Continuing with partial body evidence.",),
    )

    assert result.available is True
    assert result.value is body_result
    assert result.diagnostics == ("Pose detector unavailable.",)


def test_failed_stage_requires_actionable_diagnostic():
    with pytest.raises(ValueError, match="diagnostic"):
        PipelineStageResult(
            stage=PipelineStage.CALIBRATION,
            status=PipelineStatus.FAILED,
        )


def test_pipeline_run_result_accepts_final_measurement_and_stage_records():
    fusion_result = PipelineStageResult(
        stage=PipelineStage.FUSION,
        status=PipelineStatus.SUCCESS,
        value=make_measurement_result(),
    )
    run_result = PipelineRunResult(
        measurement_result=fusion_result.value,
        stages={PipelineStage.FUSION: fusion_result},
        diagnostics=("Measurement completed.",),
    )

    assert run_result.measurement_result.estimated_height_cm == 170.0
    assert run_result.stages[PipelineStage.FUSION].available is True


def test_pipeline_run_result_rejects_invalid_stage_values():
    with pytest.raises(TypeError, match="PipelineStageResult"):
        PipelineRunResult(
            measurement_result=None,
            stages={PipelineStage.FUSION: object()},
        )


def test_pipeline_failure_preserves_code_and_stage():
    failure = PipelineFailure(
        PipelineFailureCode.CALIBRATION_FAILED,
        "required markers were not found",
        stage=PipelineStage.CALIBRATION,
    )

    assert failure.code is PipelineFailureCode.CALIBRATION_FAILED
    assert failure.stage is PipelineStage.CALIBRATION
    assert str(failure) == "required markers were not found"


def test_stage_protocol_contracts_are_callable_with_typed_values():
    dependencies = make_dependencies()
    image = np.zeros((10, 10), dtype=np.uint8)
    calibration = object()
    body = object()

    assert isinstance(dependencies.calibration, CalibrationStage)
    assert isinstance(dependencies.quality, QualityStage)
    assert isinstance(dependencies.body_detection, BodyDetectionStage)
    assert isinstance(dependencies.estimation, EstimationStage)
    assert isinstance(dependencies.fusion, FusionStage)
    assert image.shape == (10, 10)
    assert calibration is not None
    assert body is not None


def make_calibration():
    from height_estimation.models import (
        CalibrationResult,
        DetectedMarker,
        HomographyResult,
    )

    markers = tuple(
        DetectedMarker(
            id=marker_id,
            corners=((1.0, 1.0),) * 4,
            center_x=float(marker_id),
            center_y=float(marker_id),
            area_px=4.0,
        )
        for marker_id in range(4)
    )
    return CalibrationResult(
        markers=markers,
        geometry=(),
        cm_per_pixel=1.0,
        homography=HomographyResult(
            matrix=((1.0, 0.0, 0.0),) * 3,
            reprojection_error_cm=0.0,
        ),
    )


def make_body_result(status=DetectionStatus.SUCCESS):
    detections = BodyDetections(
        keypoints={"left_heel": (5.0, 90.0, 0.9)},
        head_top=(5.0, 10.0, 0.9),
        head_bottom=(5.0, 25.0, 0.9),
        head_bbox=(0.0, 10.0, 10.0, 25.0),
        head_confidence=0.9,
    )
    return BodyDetectionResult(
        detections=detections,
        detector_results={
            "fake": DetectorResult(
                "fake",
                status,
                detections=detections,
            )
        },
        diagnostics=("fake body evidence",),
    )


class RecordingCalibration:
    def __init__(self, events):
        self.events = events

    def run(self, pipeline_input):
        self.events.append("calibration")
        return make_calibration()


class RecordingQuality:
    def __init__(self, events, *, final_quality=None):
        self.events = events
        self.final_quality = final_quality or make_quality()

    def run(self, image, *, calibration, body=None):
        self.events.append("quality_reassessment" if body else "quality_assessment")
        return self.final_quality


class RecordingBodyDetection:
    def __init__(self, events, result=None):
        self.events = events
        self.result = result or make_body_result()

    def run(self, image):
        self.events.append("body_detection")
        return self.result


class RecordingEstimation:
    def __init__(self, events, estimates=None, error=None):
        self.events = events
        self.estimates = (
            estimates
            if estimates is not None
            else (
            HeightEstimate(MeasurementMethod.GEOMETRIC, 80.0, 0.9),
            )
        )
        self.error = error

    def run(self, detections, *, calibration, quality, image_size):
        self.events.append("estimation")
        if self.error is not None:
            raise self.error
        return self.estimates


class RecordingFusion:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def run(self, estimates, *, quality):
        self.events.append("fusion")
        if self.error is not None:
            raise self.error
        return build_measurement_result(estimates, quality=quality)


def make_pipeline(tmp_path, *, quality=None, body=None, estimation=None, fusion=None):
    events = []
    image_path = tmp_path / "person.png"
    assert cv2.imwrite(str(image_path), np.full((100, 100), 255, dtype=np.uint8))
    dependencies = PipelineDependencies(
        calibration=RecordingCalibration(events),
        quality=RecordingQuality(events, final_quality=quality),
        body_detection=RecordingBodyDetection(events, result=body),
        estimation=RecordingEstimation(events, estimates=estimation),
        fusion=RecordingFusion(events, error=fusion),
    )
    return (
        MeasurementPipeline(
            dependencies=dependencies,
            config=PipelineConfig(quality_policy="continue"),
        ),
        PipelineInput(image_path=image_path, layout=make_layout()),
        events,
    )


def test_measurement_pipeline_runs_stages_in_order_and_returns_measurement(tmp_path):
    pipeline, pipeline_input, events = make_pipeline(tmp_path)

    result = pipeline.run(pipeline_input)

    assert result.measurement_result.estimated_height_cm == 80.0
    assert events == [
        "calibration",
        "quality_assessment",
        "body_detection",
        "quality_reassessment",
        "estimation",
        "fusion",
    ]
    assert set(result.stages) == {
        PipelineStage.INPUT,
        PipelineStage.CALIBRATION,
        PipelineStage.QUALITY_ASSESSMENT,
        PipelineStage.BODY_DETECTION,
        PipelineStage.QUALITY_REASSESSMENT,
        PipelineStage.ESTIMATION,
        PipelineStage.FUSION,
    }


def test_pipeline_reuses_calibration_working_image_without_redistorting(tmp_path):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    working_image = np.full((20, 30, 3), 7, dtype=np.uint8)
    image_path = tmp_path / "input.png"
    assert cv2.imwrite(str(image_path), image)
    calibration = make_calibration()
    calibration = CalibrationResult(
        markers=calibration.markers,
        geometry=calibration.geometry,
        cm_per_pixel=calibration.cm_per_pixel,
        homography=calibration.homography,
        camera_calibration=object(),
        working_image=working_image,
    )
    seen = {}

    class Calibration:
        def run(self, pipeline_input):
            del pipeline_input
            return calibration

    class Quality:
        def run(self, image_value, *, calibration, body=None):
            del calibration, body
            seen.setdefault("quality", []).append(image_value)
            return make_quality()

    class Body:
        def run(self, image_value):
            seen["body"] = image_value
            return make_body_result()

    class Estimation:
        def run(self, detections, *, calibration, quality, image_size):
            del detections, calibration, quality, image_size
            return (HeightEstimate(MeasurementMethod.GEOMETRIC, 170.0, 0.9),)

    class Fusion:
        def run(self, estimates, *, quality):
            return build_measurement_result(estimates, quality=quality)

    pipeline = MeasurementPipeline(
        dependencies=PipelineDependencies(
            calibration=Calibration(),
            quality=Quality(),
            body_detection=Body(),
            estimation=Estimation(),
            fusion=Fusion(),
        )
    )
    pipeline.run(
        PipelineInput(image_path=image_path, layout=make_layout())
    )

    assert seen["body"] is working_image
    assert all(value is working_image for value in seen["quality"])


def test_pipeline_run_result_serializes_stage_summary_as_json():
    result = PipelineRunResult(
        measurement_result=make_measurement_result(),
        stages={
            PipelineStage.FUSION: PipelineStageResult(
                stage=PipelineStage.FUSION,
                status=PipelineStatus.SUCCESS,
                value=make_measurement_result(),
            )
        },
        diagnostics=("completed",),
    )

    payload = json.loads(result.to_json(indent=2))

    assert payload["measurement_result"]["estimated_height_cm"] == 170.0
    assert payload["stages"]["fusion"]["status"] == "success"
    assert payload["diagnostics"] == ["completed"]


def test_measurement_pipeline_persists_final_and_intermediate_outputs(tmp_path):
    pipeline, pipeline_input, _ = make_pipeline(tmp_path)
    output_path = tmp_path / "results" / "measurement.json"
    intermediate_dir = tmp_path / "results" / "intermediate"
    pipeline.config = PipelineConfig(
        quality_policy="continue",
        output_path=output_path,
        intermediate_output_dir=intermediate_dir,
        write_intermediate_outputs=True,
    )

    result = pipeline.run(pipeline_input)

    final_payload = json.loads(output_path.read_text(encoding="utf-8"))
    intermediate_payload = json.loads(
        (intermediate_dir / "pipeline_stages.json").read_text(encoding="utf-8")
    )
    assert final_payload == result.measurement_result.to_dict()
    assert intermediate_payload["measurement_result"] == final_payload
    assert intermediate_payload["stages"]["persistence"]["status"] == "success"
    assert PipelineStage.PERSISTENCE in result.stages


def test_measurement_pipeline_reports_output_write_failure(tmp_path):
    blocked_path = tmp_path / "blocked"
    blocked_path.write_text("not a directory", encoding="utf-8")
    pipeline, pipeline_input, _ = make_pipeline(tmp_path)
    pipeline.config = PipelineConfig(
        quality_policy="continue",
        output_path=blocked_path / "measurement.json",
    )

    with pytest.raises(PipelineFailure) as error:
        pipeline.run(pipeline_input)

    assert error.value.code is PipelineFailureCode.OUTPUT_WRITE_FAILED
    assert error.value.stage is PipelineStage.PERSISTENCE


def test_measurement_pipeline_preserves_partial_body_detection(tmp_path):
    pipeline, pipeline_input, _ = make_pipeline(
        tmp_path,
        body=make_body_result(DetectionStatus.PARTIAL),
    )

    result = pipeline.run(pipeline_input)

    assert result.stages[PipelineStage.BODY_DETECTION].status is PipelineStatus.PARTIAL
    assert result.measurement_result is not None


def test_measurement_pipeline_rejects_failed_quality_when_policy_is_fail(tmp_path):
    failed_quality = QualityAssessment(
        passed=False,
        metrics=QualityMetrics(blur_score=0.1),
        recommendations=("Image is too blurry",),
    )
    pipeline, pipeline_input, events = make_pipeline(
        tmp_path,
        quality=failed_quality,
    )
    pipeline.config = PipelineConfig(quality_policy="fail")

    with pytest.raises(PipelineFailure) as error:
        pipeline.run(pipeline_input)

    assert error.value.code is PipelineFailureCode.QUALITY_FAILED
    assert error.value.stage is PipelineStage.QUALITY_REASSESSMENT
    assert events == ["calibration", "quality_assessment", "body_detection", "quality_reassessment"]


def test_measurement_pipeline_reports_calibration_failure(tmp_path):
    pipeline, pipeline_input, _ = make_pipeline(tmp_path)
    pipeline.dependencies = PipelineDependencies(
        calibration=RecordingCalibration([]),
        quality=RecordingQuality([]),
        body_detection=RecordingBodyDetection([]),
        estimation=RecordingEstimation([]),
        fusion=RecordingFusion([]),
    )

    class FailingCalibration:
        def run(self, pipeline_input):
            raise ValueError("missing required ArUco markers")

    pipeline.dependencies = PipelineDependencies(
        calibration=FailingCalibration(),
        quality=pipeline.dependencies.quality,
        body_detection=pipeline.dependencies.body_detection,
        estimation=pipeline.dependencies.estimation,
        fusion=pipeline.dependencies.fusion,
    )

    with pytest.raises(PipelineFailure) as error:
        pipeline.run(pipeline_input)

    assert error.value.code is PipelineFailureCode.CALIBRATION_FAILED
    assert error.value.stage is PipelineStage.CALIBRATION


def test_measurement_pipeline_reports_estimator_failure(tmp_path):
    pipeline, pipeline_input, _ = make_pipeline(
        tmp_path,
        estimation=(),
    )

    with pytest.raises(PipelineFailure) as error:
        pipeline.run(pipeline_input)

    assert error.value.code is PipelineFailureCode.ESTIMATOR_UNAVAILABLE
    assert error.value.stage is PipelineStage.ESTIMATION


def test_measurement_pipeline_reports_fusion_failure(tmp_path):
    pipeline, pipeline_input, _ = make_pipeline(
        tmp_path,
        fusion=ValueError("fusion contract failed"),
    )

    with pytest.raises(PipelineFailure) as error:
        pipeline.run(pipeline_input)

    assert error.value.code is PipelineFailureCode.FUSION_FAILED
    assert error.value.stage is PipelineStage.FUSION


def test_default_pipeline_dependencies_merge_optional_adapter_outputs():
    mask = np.zeros((12, 10), dtype=np.uint8)
    mask[1:9, 3:7] = 1
    constructions = {
        "pose": 0,
        "head": 0,
        "segmentation": 0,
        "hands": 0,
    }

    class StubPoseDetector:
        def detect(self, image):
            return {
                "shoulder_width": {
                    "landmark_11": {"x": 0.3, "y": 0.2, "visibility": 0.9},
                    "landmark_12": {"x": 0.7, "y": 0.2, "visibility": 0.9},
                },
                "upper_leg_length": {
                    "left": {
                        "landmark_23": {"x": 0.38, "y": 0.55},
                        "landmark_25": {"x": 0.38, "y": 0.72},
                    },
                    "right": {
                        "landmark_24": {"x": 0.62, "y": 0.55},
                        "landmark_26": {"x": 0.62, "y": 0.72},
                    },
                },
                "lower_leg_length": {
                    "left": {
                        "landmark_25": {"x": 0.38, "y": 0.72},
                        "landmark_27": {"x": 0.38, "y": 0.9},
                    },
                    "right": {
                        "landmark_26": {"x": 0.62, "y": 0.72},
                        "landmark_28": {"x": 0.62, "y": 0.9},
                    },
                },
                "heel_landmarks": {
                    "left": {"x": 0.38, "y": 0.95, "confidence": 0.8},
                    "right": {"x": 0.62, "y": 0.95, "confidence": 0.8},
                },
            }

    class StubHeadDetector:
        def detect(self, image):
            return {
                "head_detected": True,
                "head_bbox_pixels": {
                    "x1": 18,
                    "y1": 6,
                    "x2": 30,
                    "y2": 20,
                },
                "head_top_y_pixels": 5,
                "confidence": 0.88,
            }

    class StubSegmenter:
        def segment(self, image, *, return_mask=False):
            assert return_mask is True
            return {
                "hair_length": {
                    "top": {"y": 0.08},
                    "bottom": {"y": 0.32},
                },
                "mask": mask,
            }

    class StubHandDetector:
        def detect(self, image):
            return {
                "hands": [
                    {
                        "handedness": "Left",
                        "confidence": 0.83,
                        "hand_length": {
                            "landmark_0": {"x": 0.2, "y": 0.65},
                            "landmark_12": {"x": 0.25, "y": 0.55},
                        },
                    },
                    {
                        "handedness": "Right",
                        "confidence": 0.81,
                        "hand_length": {
                            "landmark_0": {"x": 0.8, "y": 0.65},
                            "landmark_12": {"x": 0.75, "y": 0.55},
                        },
                    },
                ]
            }

    def pose_factory():
        constructions["pose"] += 1
        return StubPoseDetector()

    def head_factory():
        constructions["head"] += 1
        return StubHeadDetector()

    def segmentation_factory():
        constructions["segmentation"] += 1
        return StubSegmenter()

    def hand_factory():
        constructions["hands"] += 1
        return StubHandDetector()

    config = PipelineConfig(
        use_person_fallback=False,
        body_detection=BodyDetectionConfig(
            pose_detector=ViTPoseDetectorAdapter(pose_factory),
            head_detector=VGGHeadsDetectorAdapter(head_factory),
            segmentation_detector=MediaPipeSegmentationAdapter(segmentation_factory),
            hand_detector=MediaPipeHandDetectorAdapter(hand_factory),
        ),
    )
    dependencies = default_pipeline_dependencies(config)
    image = np.zeros((64, 48, 3), dtype=np.uint8)

    first = dependencies.body_detection.run(image)
    second = dependencies.body_detection.run(image)

    assert first.status is DetectionStatus.SUCCESS
    assert second.status is DetectionStatus.SUCCESS
    assert constructions == {
        "pose": 1,
        "head": 1,
        "segmentation": 1,
        "hands": 1,
    }
    assert first.detections.keypoints["left_hip"].coordinate_system == "normalized"
    assert first.detections.keypoints["right_heel"].y == pytest.approx(0.95)
    assert first.detections.head_bbox == (18.0, 6.0, 30.0, 20.0)
    assert first.detections.head_top.coordinate_system == "pixel"
    assert first.detections.head_bottom.coordinate_system == "pixel"
    assert np.array_equal(first.detections.segmentation_mask, mask)
    assert first.detections.hair_top.y == pytest.approx(0.08)
    assert first.detections.handedness == ("left", "right")
    assert first.detections.hand_confidences == (0.83, 0.81)
    assert first.detector_results["segmentation"].metadata["mask_shape"] == (12, 10)
    assert first.detector_results["hands"].metadata["hand_count"] == 2


def test_default_pipeline_dependencies_preserve_partial_optional_failures():
    class StubPoseDetector:
        def detect(self, image):
            return {
                "keypoints": {
                    "left_hip": {"x": 0.4, "y": 0.5},
                    "left_heel": {"x": 0.4, "y": 0.95},
                }
            }

    class BrokenHeadDetector:
        def detect(self, image):
            raise RuntimeError("head model runtime failure")

    class PerspectiveHandDetector:
        def detect(self, image):
            return {
                "handedness_convention": "image",
                "hands": [
                    {
                        "handedness": "Left",
                        "hand_length": ((0.2, 0.6), (0.24, 0.5)),
                    }
                ],
            }

    config = PipelineConfig(
        use_person_fallback=False,
        body_detection=BodyDetectionConfig(
            pose_detector=ViTPoseDetectorAdapter(lambda: StubPoseDetector()),
            head_detector=VGGHeadsDetectorAdapter(lambda: BrokenHeadDetector()),
            segmentation_detector=MediaPipeSegmentationAdapter(
                module_name="height_estimation._missing_optional_segmenter"
            ),
            hand_detector=MediaPipeHandDetectorAdapter(lambda: PerspectiveHandDetector()),
        ),
    )
    dependencies = default_pipeline_dependencies(config)

    result = dependencies.body_detection.run(np.zeros((32, 24, 3), dtype=np.uint8))

    assert result.status is DetectionStatus.PARTIAL
    assert result.success is True
    assert "left_hip" in result.detections.keypoints
    assert result.detector_results["pose"].status is DetectionStatus.SUCCESS
    assert result.detector_results["head"].status is DetectionStatus.FAILED
    assert result.detector_results["segmentation"].status is DetectionStatus.UNAVAILABLE
    assert result.detector_results["hands"].status is DetectionStatus.FAILED
    assert any("runtime failure" in item for item in result.diagnostics)
    assert any("missing_optional_segmenter" in item for item in result.diagnostics)
    assert any("image-perspective" in item for item in result.diagnostics)


def test_body_detection_import_does_not_load_optional_backends():
    script = "\n".join(
        [
            "import json",
            "import sys",
            "import height_estimation.body_detection",
            "heavy_roots = {'mediapipe', 'torch', 'torchvision', 'easy_ViTPose', 'huggingface_hub'}",
            "loaded = sorted(name for name in sys.modules if name.split('.')[0] in heavy_roots)",
            "print(json.dumps(loaded))",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    loaded_modules = json.loads(completed.stdout.strip() or "[]")

    assert loaded_modules == []