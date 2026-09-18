from math import degrees
from typing import Mapping

import cv2
import numpy as np

from .advanced_models import (
    Landmark,
    QualityAssessment,
    QualityLevel,
    QualityMetrics,
)


def _point(value: Landmark | Mapping[str, float] | tuple[float, ...]) -> tuple[float, float]:
    landmark = Landmark.from_value(value)
    return landmark.x, landmark.y


class AcquisitionQualityGate:
    def __init__(
        self,
        *,
        min_marker_visibility: float = 0.75,
        max_pose_severity: float = 0.5,
        min_blur_score: float = 0.6,
        min_occlusion_score: float = 0.7,
    ) -> None:
        self.min_marker_visibility = min_marker_visibility
        self.max_pose_severity = max_pose_severity
        self.min_blur_score = min_blur_score
        self.min_occlusion_score = min_occlusion_score

    @staticmethod
    def blur_score(image: np.ndarray, sharpness_threshold: float = 100.0) -> float:
        if not isinstance(image, np.ndarray) or image.size == 0:
            return 0.0
        if not np.isfinite(sharpness_threshold) or sharpness_threshold <= 0:
            return 0.0
        if image.ndim == 3:
            if image.shape[2] != 3:
                return 0.0
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            return 0.0

        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not np.isfinite(variance):
            return 0.0
        return float(np.clip(variance / sharpness_threshold, 0.0, 1.0))

    @staticmethod
    def marker_visibility(
        detected_markers: int,
        expected_markers: int = 4,
    ) -> float:
        if expected_markers <= 0:
            return 0.0
        return float(
            np.clip(detected_markers / expected_markers, 0.0, 1.0)
        )

    @staticmethod
    def pose_severity(keypoints: Mapping[str, object]) -> float:
        scores: list[float] = []
        for side in ("left", "right"):
            names = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
            if all(name in keypoints for name in names):
                try:
                    hip = np.array(_point(keypoints[names[0]]), dtype=float)
                    knee = np.array(_point(keypoints[names[1]]), dtype=float)
                    ankle = np.array(_point(keypoints[names[2]]), dtype=float)
                except (TypeError, ValueError):
                    continue
                first = hip - knee
                second = ankle - knee
                denominator = np.linalg.norm(first) * np.linalg.norm(second)
                if denominator > 0 and np.isfinite(denominator):
                    angle = degrees(
                        np.arccos(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
                    )
                    if np.isfinite(angle):
                        scores.append(float(np.clip((180.0 - angle) / 45.0, 0.0, 1.0)))

        if "head_top" in keypoints and "head_bottom" in keypoints:
            try:
                top_x, top_y = _point(keypoints["head_top"])
                bottom_x, bottom_y = _point(keypoints["head_bottom"])
                dx = abs(top_x - bottom_x)
                dy = abs(top_y - bottom_y)
                if dx or dy:
                    tilt = degrees(np.arctan2(dx, dy))
                    if np.isfinite(tilt):
                        scores.append(float(np.clip(tilt / 30.0, 0.0, 1.0)))
            except (TypeError, ValueError):
                pass

        return float(np.mean(scores)) if scores else 0.5

    @staticmethod
    def occlusion_score(keypoints: Mapping[str, object]) -> float:
        confidences: list[float] = []
        for value in keypoints.values():
            if isinstance(value, Landmark):
                confidence = value.visibility
            elif isinstance(value, Mapping):
                confidence = value.get(
                    "visibility",
                    value.get("confidence", value.get("score")),
                )
            elif isinstance(value, (tuple, list)) and len(value) >= 3:
                confidence = value[2]
            else:
                confidence = None
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                continue
            if np.isfinite(confidence):
                confidences.append(float(np.clip(confidence, 0.0, 1.0)))
        return float(np.mean(confidences)) if confidences else 0.5

    @staticmethod
    def segmentation_occlusion_score(
        segmentation_mask: np.ndarray,
        body_bbox: tuple[float, float, float, float],
    ) -> float:
        try:
            mask = np.asarray(segmentation_mask)
            bbox = tuple(float(value) for value in body_bbox)
        except (TypeError, ValueError):
            return 0.0
        if mask.ndim < 2 or mask.size == 0 or len(bbox) != 4:
            return 0.0
        if not all(np.isfinite(value) for value in bbox):
            return 0.0

        height, width = mask.shape[:2]
        x1, y1, x2, y2 = bbox
        left = max(0, min(width, int(x1)))
        top = max(0, min(height, int(y1)))
        right = max(0, min(width, int(x2)))
        bottom = max(0, min(height, int(y2)))
        if right <= left or bottom <= top:
            return 0.0

        region = mask[top:bottom, left:right]
        if region.size == 0:
            return 0.0
        return float(np.count_nonzero(region) / region.size)

    def evaluate(
        self,
        image: np.ndarray,
        *,
        detected_markers: int,
        expected_markers: int = 4,
        keypoints: Mapping[str, object] | None = None,
        segmentation_mask: np.ndarray | None = None,
        body_bbox: tuple[float, float, float, float] | None = None,
    ) -> QualityAssessment:
        keypoints = keypoints or {}
        if segmentation_mask is not None and body_bbox is not None:
            occlusion_score = self.segmentation_occlusion_score(
                segmentation_mask,
                body_bbox,
            )
        else:
            occlusion_score = self.occlusion_score(keypoints)
        metrics = QualityMetrics(
            marker_visibility=self.marker_visibility(
                detected_markers,
                expected_markers,
            ),
            pose_severity=self.pose_severity(keypoints),
            blur_score=self.blur_score(image),
            occlusion_score=occlusion_score,
        )
        recommendations = []
        if metrics.blur_score < self.min_blur_score:
            recommendations.append("Image is too blurry")
        if metrics.marker_visibility < self.min_marker_visibility:
            recommendations.append("Required ArUco markers are not all visible")
        if metrics.pose_severity > self.max_pose_severity:
            recommendations.append("Subject posture is too bent or tilted")
        if metrics.occlusion_score < self.min_occlusion_score:
            recommendations.append("Body landmarks are partially occluded")
        passed = not recommendations
        return QualityAssessment(passed, metrics, tuple(recommendations))