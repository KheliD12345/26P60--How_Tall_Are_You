"""ViTPose whole-body adapter built on easy_ViTPose."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from .config_model import (
    DEFAULT_VITPOSE_MODEL,
    DEFAULT_YOLO_MODEL,
    get_model_paths,
)


class PoseDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_VITPOSE_MODEL,
        yolo_model: str = DEFAULT_YOLO_MODEL,
        device: str | None = None,
        yolo_size: int = 320,
    ) -> None:
        from easy_ViTPose import VitInference

        vitpose_path, yolo_path = get_model_paths(model_name, yolo_model)
        self.model: Any = VitInference(
            vitpose_path,
            yolo_path,
            model_name=model_name,
            yolo_size=yolo_size,
            is_video=False,
            device=device,
        )

    @staticmethod
    def _select_main_person(
        people: Mapping[object, object],
    ) -> np.ndarray | None:
        selected = None
        largest_area = 0.0
        for values in people.values():
            keypoints = np.asarray(values)
            if keypoints.ndim != 2 or keypoints.shape[1] < 3:
                raise ValueError("ViTPose returned malformed keypoints")
            visible = keypoints[keypoints[:, 2] > 0.3]
            if len(visible) < 5:
                continue
            y_coordinates, x_coordinates = visible[:, 0], visible[:, 1]
            area = (
                (x_coordinates.max() - x_coordinates.min())
                * (y_coordinates.max() - y_coordinates.min())
            )
            if area > largest_area:
                largest_area = float(area)
                selected = keypoints
        return selected

    @staticmethod
    def _landmark(
        keypoints: np.ndarray,
        index: int,
        height: int,
        width: int,
    ) -> dict[str, float]:
        y_px, x_px, confidence = keypoints[index]
        return {
            "x": float(x_px / width),
            "y": float(y_px / height),
            "z": 0.0,
            "visibility": float(confidence),
        }

    def detect(self, image: np.ndarray) -> dict[str, object]:
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        people = self.model.inference(image_rgb)
        if not people:
            return {}

        person = self._select_main_person(people)
        if person is None:
            return {}
        points = {
            index: self._landmark(person, index, height, width)
            for index in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 22)
        }
        return {
            "shoulder_width": {
                "landmark_11": points[5],
                "landmark_12": points[6],
            },
            "hip_width": {
                "landmark_23": points[11],
                "landmark_24": points[12],
            },
            "upper_arm_length": {
                "left": {"landmark_11": points[5], "landmark_13": points[7]},
                "right": {"landmark_12": points[6], "landmark_14": points[8]},
            },
            "forearm_length": {
                "left": {"landmark_13": points[7], "landmark_15": points[9]},
                "right": {"landmark_14": points[8], "landmark_16": points[10]},
            },
            "upper_leg_length": {
                "left": {"landmark_23": points[11], "landmark_25": points[13]},
                "right": {"landmark_24": points[12], "landmark_26": points[14]},
            },
            "lower_leg_length": {
                "left": {"landmark_25": points[13], "landmark_27": points[15]},
                "right": {"landmark_26": points[14], "landmark_28": points[16]},
            },
            "shoulder_to_waist": {
                "left": {"landmark_11": points[5], "landmark_23": points[11]},
                "right": {"landmark_12": points[6], "landmark_24": points[12]},
            },
            "heel_landmarks": {
                "left": points[19],
                "right": points[22],
            },
        }

    def close(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()
