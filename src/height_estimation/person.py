from pathlib import Path

import cv2
import numpy as np

from .models import PersonEndpoints


def select_person_candidate(
    candidates: list[tuple[int, int, int, int, float]],
) -> tuple[int, int, int, int, float] | None:
    useful_candidates = [
        candidate
        for candidate in candidates
        if candidate[2] >= 40
        and candidate[3] >= 80
        and candidate[3] / candidate[2] >= 1.4
    ]
    if not useful_candidates:
        return None
    return max(useful_candidates, key=lambda candidate: candidate[2] * candidate[3])


def detect_person_endpoints(
    image_path: str | Path,
) -> PersonEndpoints | None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")

    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = detector.detectMultiScale(
        image,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    candidates = []
    for box, weight in zip(boxes, weights):
        x, y, width, height = (int(value) for value in box)
        candidates.append((x, y, width, height, float(weight)))

    candidate = select_person_candidate(candidates)
    if candidate is None:
        return None

    x, y, width, height, score = candidate
    image_height, image_width = image.shape[:2]
    left = max(0, min(x, image_width - 1))
    top = max(0, min(y, image_height - 1))
    right = max(left, min(x + width - 1, image_width - 1))
    bottom = max(top, min(y + height - 1, image_height - 1))
    clipped_width = right - left + 1
    clipped_height = bottom - top + 1
    center_x = left + clipped_width / 2

    return PersonEndpoints(
        box=(left, top, clipped_width, clipped_height),
        top_of_head=(center_x, float(top)),
        bottom_of_feet=(center_x, float(bottom)),
        score=score,
    )