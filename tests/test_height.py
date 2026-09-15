import pytest

from height_estimation.geometry import transform_person_endpoints
from height_estimation.height import (
    calculate_perspective_height_cm,
    validate_perspective_result,
)
from height_estimation.models import PersonEndpoints


def test_transforms_person_endpoints_to_physical_coordinates():
    person = PersonEndpoints(
        box=(10, 20, 80, 220),
        top_of_head=(20.0, 40.0),
        bottom_of_feet=(20.0, 240.0),
        score=1.0,
    )
    homography = (
        (0.5, 0.0, 0.0),
        (0.0, 0.25, 0.0),
        (0.0, 0.0, 1.0),
    )

    head, feet = transform_person_endpoints(person, homography)

    assert head == pytest.approx((10.0, 10.0))
    assert feet == pytest.approx((10.0, 60.0))


def test_calculates_height_between_transformed_endpoints():
    head = (35.0, 160.0)
    feet = (42.0, 0.0)

    assert calculate_perspective_height_cm(head, feet) == pytest.approx(160.0)


def test_rejects_invalid_perspective_endpoints():
    with pytest.raises(ValueError, match="finite"):
        calculate_perspective_height_cm((float("nan"), 160.0), (42.0, 0.0))

    with pytest.raises(ValueError, match="different heights"):
        calculate_perspective_height_cm((35.0, 20.0), (42.0, 20.0))


def test_rejects_unreliable_perspective_result():
    with pytest.raises(ValueError, match="too high"):
        validate_perspective_result(
            (35.0, 160.0),
            (42.0, 0.0),
            reprojection_error_cm=2.1,
        )

    validate_perspective_result(
        (35.0, 160.0),
        (42.0, 0.0),
        reprojection_error_cm=2.0,
    )