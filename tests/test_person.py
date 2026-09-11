from pathlib import Path

import cv2
import numpy as np

from height_estimation.person import detect_person_endpoints, select_person_candidate


def test_selects_largest_useful_full_body_candidate():
    candidates = [
        (20, 30, 30, 50, 2.0),
        (100, 40, 120, 300, 1.2),
    ]

    assert select_person_candidate(candidates) == (100, 40, 120, 300, 1.2)


def test_returns_none_when_no_useful_candidate_exists():
    candidates = [(20, 30, 30, 50, 2.0)]

    assert select_person_candidate(candidates) is None


def test_no_person_returns_none(tmp_path, monkeypatch):
    image_path = tmp_path / "empty.png"
    cv2.imwrite(str(image_path), np.zeros((200, 200, 3), dtype=np.uint8))

    class EmptyDetector:
        def setSVMDetector(self, detector):
            pass

        def detectMultiScale(self, image, **kwargs):
            return (), ()

    monkeypatch.setattr("height_estimation.person.cv2.HOGDescriptor", EmptyDetector)

    assert detect_person_endpoints(image_path) is None


def test_endpoints_are_ordered_and_inside_image_bounds(tmp_path, monkeypatch):
    image_path = tmp_path / "person.png"
    cv2.imwrite(str(image_path), np.zeros((200, 160, 3), dtype=np.uint8))

    class CandidateDetector:
        def setSVMDetector(self, detector):
            pass

        def detectMultiScale(self, image, **kwargs):
            return np.array([[-20, -10, 100, 250]]), np.array([1.5])

    monkeypatch.setattr(
        "height_estimation.person.cv2.HOGDescriptor",
        CandidateDetector,
    )

    result = detect_person_endpoints(image_path)

    assert result is not None
    assert result.top_of_head[1] < result.bottom_of_feet[1]
    assert 0 <= result.top_of_head[0] < 160
    assert 0 <= result.top_of_head[1] < 200
    assert 0 <= result.bottom_of_feet[0] < 160
    assert 0 <= result.bottom_of_feet[1] < 200