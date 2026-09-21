from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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

from .advanced_models import (
    BodyDetections,
    HeightEstimate,
    MeasurementResult,
    QualityAssessment,
)
from .body_detection import BodyDetectionResult
from .models import CalibrationResult, MarkerLayout


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

    def __post_init__(self) -> None:
        if self.quality_policy not in {"fail", "continue"}:
            raise ValueError("quality policy must be fail or continue")
        if not isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must be between zero and one")
        if not isinstance(self.use_person_fallback, bool):
            raise ValueError("use_person_fallback must be a boolean")


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