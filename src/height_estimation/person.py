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


def refine_person_box(
    image: np.ndarray,
    candidate: tuple[int, int, int, int, float],
) -> tuple[int, int, int, int]:
    x, y, width, height, _ = candidate
    image_height, image_width = image.shape[:2]
    right = min(image_width, x + width)
    bottom = min(image_height, y + height)
    crop = image[max(0, y) : bottom, max(0, x) : right]
    if crop.size == 0:
        return x, y, width, height

    crop_height, crop_width = crop.shape[:2]
    mask = np.full((crop_height, crop_width), cv2.GC_BGD, dtype=np.uint8)
    inner_left = max(1, crop_width // 5)
    inner_right = min(crop_width - 1, crop_width * 4 // 5)
    inner_top = max(1, crop_height // 20)
    inner_bottom = min(crop_height - 1, crop_height * 19 // 20)
    mask[inner_top:inner_bottom, inner_left:inner_right] = cv2.GC_PR_FGD
    cv2.grabCut(crop, mask, None, np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)

    foreground = np.uint8(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    if count <= 1:
        return x, y, width, height

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == largest
    points = cv2.findNonZero(np.uint8(component))
    if points is None:
        return x, y, width, height

    refined_x, refined_y, refined_width, refined_height = cv2.boundingRect(points)
    if refined_height < height * 0.3:
        return x, y, width, height
    return (
        max(0, x + refined_x),
        max(0, y + refined_y),
        refined_width,
        refined_height,
    )


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
    x, y, width, height = refine_person_box(image, candidate)
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