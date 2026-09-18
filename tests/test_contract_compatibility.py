import json
from math import inf, nan

import pytest

from height_estimation.advanced_models import (
    AdvancedMeasurementResult,
    HeightEstimate,
    MeasurementMethod,
    MeasurementResult,
    QualityAssessment,
    QualityLevel,
    QualityMetrics,
)
from height_estimation.models import (
    CalibrationResult,
    HomographyResult,
    PersonEndpoints,
)
from height_estimation.quality import QualityLevel as GateQualityLevel


def make_quality() -> QualityAssessment:
    return QualityAssessment(
        passed=True,
        metrics=QualityMetrics(marker_visibility=1.0),
    )


def test_contract_imports_preserve_public_names_and_aliases():
    assert AdvancedMeasurementResult is MeasurementResult
    assert GateQualityLevel is QualityLevel
    assert MeasurementMethod.GEOMETRIC.value == "geometric"
    assert MeasurementMethod.SKELETON.value == "skeleton"
    assert MeasurementMethod.HEAD_BBOX.value == "head_bbox"
    assert MeasurementMethod.SMPL_BASED.value == "smpl_based"
    assert MeasurementMethod.ANTHROPOMETRIC.value == "anthropometric"


def test_measurement_result_json_has_stable_complete_shape():
    result = MeasurementResult(
        estimated_height_cm=171.25,
        uncertainty_range=(169.0, 173.5),
        quality=make_quality(),
        method_estimates=(
            HeightEstimate(
                method="geometric",
                height_cm=171.2,
                confidence=0.9,
                metadata={"source": "calibration"},
            ),
        ),
        fusion_weights={"geometric": 1.0},
        measurements={"head_to_heel_cm": 171.2},
        diagnostics=("Single method available",),
        warnings=("No hair landmarks",),
    )

    payload = result.to_dict()

    assert list(payload) == [
        "estimated_height_cm",
        "uncertainty_range",
        "quality",
        "method_estimates",
        "fusion_weights",
        "measurements",
        "diagnostics",
        "warnings",
    ]
    assert json.loads(result.to_json()) == payload
    assert result.to_json() == result.to_json()
    assert payload["method_estimates"][0]["method"] == "geometric"
    assert payload["method_estimates"][0]["metadata"] == {
        "source": "calibration"
    }


def test_measurement_result_serialises_missing_optional_data_as_empty():
    payload = MeasurementResult(
        estimated_height_cm=170.0,
        uncertainty_range=(170.0, 170.0),
        quality=make_quality(),
    ).to_dict()

    assert payload["method_estimates"] == []
    assert payload["fusion_weights"] == {}
    assert payload["measurements"] == {}
    assert payload["diagnostics"] == []
    assert payload["warnings"] == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"estimated_height_cm": 0.0},
        {"estimated_height_cm": inf},
        {"uncertainty_range": (175.0, 170.0)},
        {"uncertainty_range": (-1.0, 170.0)},
        {"estimated_height_cm": 180.0},
        {"fusion_weights": {"geometric": -0.1}},
        {"measurements": {"head_to_heel_cm": nan}},
    ],
)
def test_measurement_result_rejects_invalid_values(overrides):
    values = {
        "estimated_height_cm": 170.0,
        "uncertainty_range": (169.0, 171.0),
        "quality": make_quality(),
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        MeasurementResult(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "unsupported"},
        {"height_cm": 0.0},
        {"confidence": 1.1},
    ],
)
def test_height_estimate_rejects_invalid_values(overrides):
    values = {
        "method": MeasurementMethod.GEOMETRIC,
        "height_cm": 170.0,
        "confidence": 0.8,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        HeightEstimate(**values)


def test_height_estimate_normalises_nested_json_metadata():
    estimate = HeightEstimate(
        method=MeasurementMethod.GEOMETRIC,
        height_cm=170.0,
        confidence=0.8,
        metadata={"sources": ("aruco", "pose"), "details": {"version": 1}},
    )

    assert estimate.to_dict()["metadata"] == {
        "sources": ["aruco", "pose"],
        "details": {"version": 1},
    }


def test_height_estimate_rejects_non_string_metadata_keys():
    with pytest.raises(ValueError):
        HeightEstimate(
            method=MeasurementMethod.GEOMETRIC,
            height_cm=170.0,
            confidence=0.8,
            metadata={1: "aruco"},
        )


def test_calibration_serialisation_remains_compatible():
    person = PersonEndpoints(
        box=(10, 20, 80, 180),
        top_of_head=(50.0, 20.0),
        bottom_of_feet=(50.0, 199.0),
        score=1.0,
    )
    calibration = CalibrationResult(
        markers=(),
        geometry=(),
        cm_per_pixel=0.1,
        homography=HomographyResult(
            matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            reprojection_error_cm=0.0,
        ),
        person=person,
    )

    payload = calibration.to_dict()

    assert set(payload) == {"markers", "geometry", "scale", "homography", "person"}
    assert payload["person"]["height_cm"] == 17.9
    assert json.loads(json.dumps(payload)) == payload