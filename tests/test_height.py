import pytest

from height_estimation.geometry import transform_person_endpoints
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