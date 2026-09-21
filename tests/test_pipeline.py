from pathlib import Path

import numpy as np
import pytest
import cv2

from height_estimation.advanced_models import (
    BodyDetections,
    HeightEstimate,
    MeasurementMethod,
    MeasurementResult,
    QualityAssessment,
    QualityMetrics,
)
from height_estimation.body_detection import (
    BodyDetectionResult,
    DetectionStatus,
    DetectorResult,
)
from height_estimation.models import MarkerLayout, MarkerPosition
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