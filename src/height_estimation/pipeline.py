from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from typing import (
    Generic,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import cv2

from .advanced_models import (
    BodyDetections,
    HeightEstimate,
    MeasurementResult,
    QualityAssessment,
)
from .body_detection import (
    BodyDetectionOrchestrator,
    BodyDetectionResult,
    DetectionStatus,
)
from .calibration import calibrate_image
from .camera import undistort_image
from .estimators import estimate_independent_heights
from .fusion import build_measurement_result
from .models import CalibrationResult, CameraCalibration, MarkerLayout
from .quality import AcquisitionQualityGate


class PipelineStage(str, Enum):
    INPUT = "input"
    CALIBRATION = "calibration"
    QUALITY_ASSESSMENT = "quality_assessment"
    BODY_DETECTION = "body_detection"
    QUALITY_REASSESSMENT = "quality_reassessment"
    ESTIMATION = "estimation"
    FUSION = "fusion"
    PERSISTENCE = "persistence"


PIPELINE_STAGE_ORDER = (
    PipelineStage.INPUT,
    PipelineStage.CALIBRATION,
    PipelineStage.QUALITY_ASSESSMENT,
    PipelineStage.BODY_DETECTION,
    PipelineStage.QUALITY_REASSESSMENT,
    PipelineStage.ESTIMATION,
    PipelineStage.FUSION,
    PipelineStage.PERSISTENCE,
)


class PipelineStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class PipelineFailureCode(str, Enum):
    INPUT_MISSING = "input_missing"
    IMAGE_UNREADABLE = "image_unreadable"
    CALIBRATION_FAILED = "calibration_failed"
    QUALITY_FAILED = "quality_failed"
    BODY_DETECTION_UNAVAILABLE = "body_detection_unavailable"
    NO_PERSON_DETECTED = "no_person_detected"
    ESTIMATOR_UNAVAILABLE = "estimator_unavailable"
    FUSION_FAILED = "fusion_failed"
    OUTPUT_WRITE_FAILED = "output_write_failed"


@dataclass(frozen=True)
class PipelineInput:
    image_path: Path
    layout: MarkerLayout
    camera_calibration_path: Path | None = None

    def __post_init__(self) -> None:
        image_path = Path(self.image_path)
        if not str(image_path).strip():
            raise ValueError("pipeline image path must not be empty")
        if not isinstance(self.layout, MarkerLayout):
            raise TypeError("pipeline layout must be a MarkerLayout")
        camera_path = (
            None
            if self.camera_calibration_path is None
            else Path(self.camera_calibration_path)
        )
        object.__setattr__(self, "image_path", image_path)
        object.__setattr__(self, "camera_calibration_path", camera_path)


@dataclass(frozen=True)
class PipelineConfig:
    quality_policy: Literal["fail", "continue"] = "fail"
    confidence_level: float = 0.95
    use_person_fallback: bool = True
    output_path: Path | None = None
    intermediate_output_dir: Path | None = None
    write_intermediate_outputs: bool = False

    def __post_init__(self) -> None:
        if self.quality_policy not in {"fail", "continue"}:
            raise ValueError("quality policy must be fail or continue")
        if not isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must be between zero and one")
        if not isinstance(self.use_person_fallback, bool):
            raise ValueError("use_person_fallback must be a boolean")
        output_path = None if self.output_path is None else Path(self.output_path)
        intermediate_dir = (
            None
            if self.intermediate_output_dir is None
            else Path(self.intermediate_output_dir)
        )
        if output_path is not None and not str(output_path).strip():
            raise ValueError("output_path must not be empty")
        if intermediate_dir is not None and not str(intermediate_dir).strip():
            raise ValueError("intermediate_output_dir must not be empty")
        if not isinstance(self.write_intermediate_outputs, bool):
            raise ValueError("write_intermediate_outputs must be a boolean")
        if self.write_intermediate_outputs and intermediate_dir is None:
            raise ValueError(
                "intermediate_output_dir is required when writing intermediate outputs"
            )
        object.__setattr__(self, "output_path", output_path)
        object.__setattr__(self, "intermediate_output_dir", intermediate_dir)


StageValue = TypeVar("StageValue")


@dataclass(frozen=True)
class PipelineStageResult(Generic[StageValue]):
    stage: PipelineStage
    status: PipelineStatus
    value: StageValue | None = None
    diagnostics: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", PipelineStage(self.stage))
        object.__setattr__(self, "status", PipelineStatus(self.status))
        diagnostics = tuple(self.diagnostics)
        warnings = tuple(self.warnings)
        if any(not isinstance(item, str) or not item.strip() for item in diagnostics):
            raise ValueError("pipeline diagnostics must be non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in warnings):
            raise ValueError("pipeline warnings must be non-empty strings")
        if self.status == PipelineStatus.FAILED and not diagnostics:
            raise ValueError("failed pipeline stages require a diagnostic")
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "warnings", warnings)

    @property
    def available(self) -> bool:
        return self.status in {PipelineStatus.SUCCESS, PipelineStatus.PARTIAL}

    def to_dict(self) -> dict[str, object]:
        value = self.value
        if hasattr(value, "to_dict"):
            serialised_value = value.to_dict()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            serialised_value = value
        else:
            serialised_value = None
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "value": serialised_value,
            "diagnostics": list(self.diagnostics),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PipelineRunResult:
    measurement_result: MeasurementResult | None
    stages: Mapping[PipelineStage, PipelineStageResult[object]] = field(
        default_factory=dict
    )
    diagnostics: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        stages = dict(self.stages)
        if any(not isinstance(stage, PipelineStage) for stage in stages):
            raise TypeError("pipeline stage results must use PipelineStage keys")
        if any(
            not isinstance(result, PipelineStageResult)
            or result.stage != stage
            for stage, result in stages.items()
        ):
            raise TypeError("pipeline stages must contain PipelineStageResult values")
        if any(not isinstance(item, str) or not item.strip() for item in self.diagnostics):
            raise ValueError("pipeline diagnostics must be non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.warnings):
            raise ValueError("pipeline warnings must be non-empty strings")
        if self.measurement_result is not None and not isinstance(
            self.measurement_result,
            MeasurementResult,
        ):
            raise TypeError("measurement_result must be a MeasurementResult")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "measurement_result": (
                None
                if self.measurement_result is None
                else self.measurement_result.to_dict()
            ),
            "stages": {
                stage.value: result.to_dict()
                for stage, result in self.stages.items()
            },
            "diagnostics": list(self.diagnostics),
            "warnings": list(self.warnings),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@runtime_checkable
class CalibrationStage(Protocol):
    def run(self, pipeline_input: PipelineInput) -> CalibrationResult:
        ...


@runtime_checkable
class QualityStage(Protocol):
    def run(
        self,
        image: np.ndarray,
        *,
        calibration: CalibrationResult,
        body: BodyDetectionResult | None = None,
    ) -> QualityAssessment:
        ...


@runtime_checkable
class BodyDetectionStage(Protocol):
    def run(self, image: np.ndarray) -> BodyDetectionResult:
        ...


@runtime_checkable
class EstimationStage(Protocol):
    def run(
        self,
        detections: BodyDetections,
        *,
        calibration: CalibrationResult,
        quality: QualityAssessment,
        image_size: tuple[int, int],
    ) -> Sequence[HeightEstimate]:
        ...


@runtime_checkable
class FusionStage(Protocol):
    def run(
        self,
        estimates: Sequence[HeightEstimate],
        *,
        quality: QualityAssessment,
    ) -> MeasurementResult:
        ...


@dataclass(frozen=True)
class PipelineDependencies:
    calibration: CalibrationStage
    quality: QualityStage
    body_detection: BodyDetectionStage
    estimation: EstimationStage
    fusion: FusionStage


class _CalibrationStage:
    def run(self, pipeline_input: PipelineInput) -> CalibrationResult:
        return calibrate_image(
            pipeline_input.image_path,
            pipeline_input.layout,
            camera_calibration_path=pipeline_input.camera_calibration_path,
        )


class _QualityStage:
    def __init__(self, gate: AcquisitionQualityGate | None = None) -> None:
        self.gate = gate or AcquisitionQualityGate()

    def run(
        self,
        image: np.ndarray,
        *,
        calibration: CalibrationResult,
        body: BodyDetectionResult | None = None,
    ) -> QualityAssessment:
        detections = None if body is None else body.detections
        body_bbox = None
        if (
            detections is not None
            and detections.head_bbox is not None
            and detections.head_bbox_coordinate_system == "pixel"
        ):
            body_bbox = detections.head_bbox
        return self.gate.evaluate(
            image,
            detected_markers=len(calibration.markers),
            expected_markers=len(calibration.markers),
            keypoints=(
                {} if detections is None else detections.keypoints
            ),
            segmentation_mask=(
                None if detections is None else detections.segmentation_mask
            ),
            body_bbox=body_bbox,
        )


class _BodyDetectionStage:
    def __init__(self, *, use_person_fallback: bool) -> None:
        self.orchestrator = BodyDetectionOrchestrator(
            use_person_fallback=use_person_fallback,
        )

    def run(self, image: np.ndarray) -> BodyDetectionResult:
        return self.orchestrator.detect(image)


class _EstimationStage:
    def run(
        self,
        detections: BodyDetections,
        *,
        calibration: CalibrationResult,
        quality: QualityAssessment,
        image_size: tuple[int, int],
    ) -> Sequence[HeightEstimate]:
        return estimate_independent_heights(
            detections,
            cm_per_pixel=calibration.cm_per_pixel,
            image_size=image_size,
            quality=quality,
        )


class _FusionStage:
    def __init__(self, *, confidence_level: float) -> None:
        self.confidence_level = confidence_level

    def run(
        self,
        estimates: Sequence[HeightEstimate],
        *,
        quality: QualityAssessment,
    ) -> MeasurementResult:
        return build_measurement_result(
            estimates,
            quality=quality,
            confidence_level=self.confidence_level,
        )


def default_pipeline_dependencies(
    config: PipelineConfig | None = None,
) -> PipelineDependencies:
    pipeline_config = config or PipelineConfig()
    return PipelineDependencies(
        calibration=_CalibrationStage(),
        quality=_QualityStage(),
        body_detection=_BodyDetectionStage(
            use_person_fallback=pipeline_config.use_person_fallback,
        ),
        estimation=_EstimationStage(),
        fusion=_FusionStage(
            confidence_level=pipeline_config.confidence_level,
        ),
    )


class MeasurementPipeline:
    """Coordinate image calibration, body evidence, estimation, and fusion."""

    def __init__(
        self,
        *,
        dependencies: PipelineDependencies | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.dependencies = dependencies or default_pipeline_dependencies(
            self.config,
        )

    def run(self, pipeline_input: PipelineInput) -> PipelineRunResult:
        stages: dict[PipelineStage, PipelineStageResult[object]] = {}
        image = self._load_image(pipeline_input, stages)
        calibration = self._run_calibration(pipeline_input, stages)
        working_image = image
        if calibration.camera_calibration is not None:
            working_image = undistort_image(image, calibration.camera_calibration)

        initial_quality = self._run_quality(
            working_image,
            calibration=calibration,
            body=None,
            stage=PipelineStage.QUALITY_ASSESSMENT,
            stages=stages,
        )
        del initial_quality

        body = self._run_body_detection(working_image, stages)
        if body.status == DetectionStatus.NO_PERSON:
            self._fail(
                PipelineFailureCode.NO_PERSON_DETECTED,
                "no person was detected in the input image",
                PipelineStage.BODY_DETECTION,
                stages,
            )
        if body.status in {
            DetectionStatus.UNAVAILABLE,
            DetectionStatus.FAILED,
        }:
            diagnostic = "; ".join(body.diagnostics) or (
                f"body detection is {body.status.value}"
            )
            self._fail(
                PipelineFailureCode.BODY_DETECTION_UNAVAILABLE,
                diagnostic,
                PipelineStage.BODY_DETECTION,
                stages,
            )

        quality = self._run_quality(
            working_image,
            calibration=calibration,
            body=body,
            stage=PipelineStage.QUALITY_REASSESSMENT,
            stages=stages,
        )
        if not quality.passed and self.config.quality_policy == "fail":
            self._fail(
                PipelineFailureCode.QUALITY_FAILED,
                "; ".join(quality.recommendations)
                or "acquisition quality gate failed",
                PipelineStage.QUALITY_REASSESSMENT,
                stages,
            )

        estimates = self._run_estimation(
            body.detections,
            calibration=calibration,
            quality=quality,
            image_size=(working_image.shape[0], working_image.shape[1]),
            stages=stages,
        )
        if not estimates:
            self._fail(
                PipelineFailureCode.ESTIMATOR_UNAVAILABLE,
                "no independent height estimator produced a result",
                PipelineStage.ESTIMATION,
                stages,
            )

        measurement = self._run_fusion(estimates, quality, stages)
        diagnostics = self._collect_messages(stages, "diagnostics")
        warnings = self._collect_messages(stages, "warnings")
        self._persist(
            measurement,
            stages=stages,
            diagnostics=diagnostics,
            warnings=warnings,
        )
        diagnostics = self._collect_messages(stages, "diagnostics")
        warnings = self._collect_messages(stages, "warnings")
        return PipelineRunResult(
            measurement_result=measurement,
            stages=stages,
            diagnostics=diagnostics,
            warnings=warnings,
        )

    @staticmethod
    def _load_image(
        pipeline_input: PipelineInput,
        stages: dict[PipelineStage, PipelineStageResult[object]],
    ) -> np.ndarray:
        if not pipeline_input.image_path.is_file():
            return MeasurementPipeline._fail(
                PipelineFailureCode.INPUT_MISSING,
                f"input image does not exist: {pipeline_input.image_path}",
                PipelineStage.INPUT,
                stages,
            )
        image = cv2.imread(
            str(pipeline_input.image_path),
            cv2.IMREAD_UNCHANGED,
        )
        if image is None or image.size == 0:
            return MeasurementPipeline._fail(
                PipelineFailureCode.IMAGE_UNREADABLE,
                f"input image could not be read: {pipeline_input.image_path}",
                PipelineStage.INPUT,
                stages,
            )
        stages[PipelineStage.INPUT] = PipelineStageResult(
            stage=PipelineStage.INPUT,
            status=PipelineStatus.SUCCESS,
            value=image,
        )
        return image

    def _run_calibration(
        self,
        pipeline_input: PipelineInput,
        stages: dict[PipelineStage, PipelineStageResult[object]],
    ) -> CalibrationResult:
        try:
            calibration = self.dependencies.calibration.run(pipeline_input)
        except Exception as error:
            return self._fail(
                PipelineFailureCode.CALIBRATION_FAILED,
                f"metric scene recovery failed: {error}",
                PipelineStage.CALIBRATION,
                stages,
            )
        stages[PipelineStage.CALIBRATION] = PipelineStageResult(
            stage=PipelineStage.CALIBRATION,
            status=PipelineStatus.SUCCESS,
            value=calibration,
            diagnostics=calibration.diagnostics,
        )
        return calibration

    def _run_quality(
        self,
        image: np.ndarray,
        *,
        calibration: CalibrationResult,
        body: BodyDetectionResult | None,
        stage: PipelineStage,
        stages: dict[PipelineStage, PipelineStageResult[object]],
    ) -> QualityAssessment:
        try:
            quality = self.dependencies.quality.run(
                image,
                calibration=calibration,
                body=body,
            )
        except Exception as error:
            self._fail(
                PipelineFailureCode.QUALITY_FAILED,
                f"quality assessment failed: {error}",
                stage,
                stages,
            )
        stages[stage] = PipelineStageResult(
            stage=stage,
            status=(
                PipelineStatus.SUCCESS
                if quality.passed
                else PipelineStatus.PARTIAL
            ),
            value=quality,
            warnings=quality.recommendations,
        )
        return quality

    def _run_body_detection(
        self,
        image: np.ndarray,
        stages: dict[PipelineStage, PipelineStageResult[object]],
    ) -> BodyDetectionResult:
        try:
            body = self.dependencies.body_detection.run(image)
        except Exception as error:
            self._fail(
                PipelineFailureCode.BODY_DETECTION_UNAVAILABLE,
                f"body detection failed: {error}",
                PipelineStage.BODY_DETECTION,
                stages,
            )
        status = (
            PipelineStatus.SUCCESS
            if body.status == DetectionStatus.SUCCESS
            else PipelineStatus.PARTIAL
            if body.status == DetectionStatus.PARTIAL
            else PipelineStatus.FAILED
        )
        stages[PipelineStage.BODY_DETECTION] = PipelineStageResult(
            stage=PipelineStage.BODY_DETECTION,
            status=status,
            value=body,
            diagnostics=body.diagnostics,
        )
        return body

    def _run_estimation(
        self,
        detections: BodyDetections,
        *,
        calibration: CalibrationResult,
        quality: QualityAssessment,
        image_size: tuple[int, int],
        stages: dict[PipelineStage, PipelineStageResult[object]],
    ) -> tuple[HeightEstimate, ...]:
        try:
            estimates = tuple(
                self.dependencies.estimation.run(
                    detections,
                    calibration=calibration,
                    quality=quality,
                    image_size=image_size,
                )
            )
        except Exception as error:
            return self._fail(
                PipelineFailureCode.ESTIMATOR_UNAVAILABLE,
                f"height estimation failed: {error}",
                PipelineStage.ESTIMATION,
                stages,
            )
        if any(not isinstance(estimate, HeightEstimate) for estimate in estimates):
            return self._fail(
                PipelineFailureCode.ESTIMATOR_UNAVAILABLE,
                "height estimator returned an invalid estimate",
                PipelineStage.ESTIMATION,
                stages,
            )
        stages[PipelineStage.ESTIMATION] = PipelineStageResult(
            stage=PipelineStage.ESTIMATION,
            status=(
                PipelineStatus.SUCCESS
                if estimates
                else PipelineStatus.FAILED
            ),
            value=estimates,
            diagnostics=(
                f"Produced {len(estimates)} independent height estimate(s).",
            ),
        )
        return estimates

    def _run_fusion(
        self,
        estimates: Sequence[HeightEstimate],
        quality: QualityAssessment,
        stages: dict[PipelineStage, PipelineStageResult[object]],
    ) -> MeasurementResult:
        try:
            measurement = self.dependencies.fusion.run(
                estimates,
                quality=quality,
            )
        except Exception as error:
            return self._fail(
                PipelineFailureCode.FUSION_FAILED,
                f"measurement fusion failed: {error}",
                PipelineStage.FUSION,
                stages,
            )
        stages[PipelineStage.FUSION] = PipelineStageResult(
            stage=PipelineStage.FUSION,
            status=PipelineStatus.SUCCESS,
            value=measurement,
            diagnostics=measurement.diagnostics,
            warnings=measurement.warnings,
        )
        return measurement

    def _persist(
        self,
        measurement: MeasurementResult,
        *,
        stages: dict[PipelineStage, PipelineStageResult[object]],
        diagnostics: tuple[str, ...],
        warnings: tuple[str, ...],
    ) -> None:
        output_path = self.config.output_path
        intermediate_dir = self.config.intermediate_output_dir
        if output_path is None and not self.config.write_intermediate_outputs:
            return
        written_paths: list[Path] = []
        try:
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    measurement.to_json(indent=2) + "\n",
                    encoding="utf-8",
                )
                written_paths.append(output_path)
            if self.config.write_intermediate_outputs:
                if intermediate_dir is None:
                    raise ValueError("intermediate output directory is not configured")
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                intermediate_path = intermediate_dir / "pipeline_stages.json"
                written_paths.append(intermediate_path)
            stages[PipelineStage.PERSISTENCE] = PipelineStageResult(
                stage=PipelineStage.PERSISTENCE,
                status=PipelineStatus.SUCCESS,
                diagnostics=(
                    "Wrote pipeline output to: "
                    + ", ".join(str(path) for path in written_paths),
                ),
            )
            if self.config.write_intermediate_outputs:
                intermediate_path.write_text(
                    json.dumps(
                        {
                            "measurement_result": measurement.to_dict(),
                            "stages": {
                                stage.value: result.to_dict()
                                for stage, result in stages.items()
                            },
                            "diagnostics": list(diagnostics),
                            "warnings": list(warnings),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        except (OSError, TypeError, ValueError) as error:
            self._fail(
                PipelineFailureCode.OUTPUT_WRITE_FAILED,
                f"could not write pipeline output: {error}",
                PipelineStage.PERSISTENCE,
                stages,
            )

    @staticmethod
    def _collect_messages(
        stages: Mapping[PipelineStage, PipelineStageResult[object]],
        attribute: str,
    ) -> tuple[str, ...]:
        messages: list[str] = []
        for stage in PIPELINE_STAGE_ORDER:
            result = stages.get(stage)
            if result is not None:
                messages.extend(getattr(result, attribute))
        return tuple(messages)

    @staticmethod
    def _fail(
        code: PipelineFailureCode,
        message: str,
        stage: PipelineStage,
        stages: dict[PipelineStage, PipelineStageResult[object]],
    ) -> None:
        stages[stage] = PipelineStageResult(
            stage=stage,
            status=PipelineStatus.FAILED,
            diagnostics=(message,),
        )
        raise PipelineFailure(code, message, stage=stage)


class PipelineFailure(RuntimeError):
    def __init__(
        self,
        code: PipelineFailureCode | str,
        message: str,
        *,
        stage: PipelineStage | str,
    ) -> None:
        self.code = PipelineFailureCode(code)
        self.stage = PipelineStage(stage)
        self.message = message
        super().__init__(message)