from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any

import cv2

from .height import calculate_height_cm


@dataclass(frozen=True)
class MarkerPosition:
    id: int
    name: str
    x_cm: float
    y_cm: float

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("marker IDs must be non-negative")
        if not self.name.strip():
            raise ValueError("marker names must not be empty")
        if not all(isfinite(value) for value in (self.x_cm, self.y_cm)):
            raise ValueError("marker positions must have finite coordinates")


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: tuple[tuple[float, ...], ...]
    distortion_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.camera_matrix) != 3 or any(
            len(row) != 3 for row in self.camera_matrix
        ):
            raise ValueError("camera matrix must have shape 3x3")
        if len(self.distortion_coefficients) not in (4, 5, 8, 12, 14):
            raise ValueError("distortion coefficients have an unsupported shape")
        values = [value for row in self.camera_matrix for value in row]
        if not all(isfinite(value) for value in values):
            raise ValueError("camera calibration values must be finite")
        if not all(isfinite(value) for value in self.distortion_coefficients):
            raise ValueError("camera calibration values must be finite")
        if self.camera_matrix[0][0] <= 0 or self.camera_matrix[1][1] <= 0:
            raise ValueError("camera focal lengths must be greater than zero")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        if not isinstance(data, dict):
            raise ValueError("camera calibration must be a JSON object")
        try:
            matrix = tuple(
                tuple(float(value) for value in row)
                for row in data["camera_matrix"]
            )
            coefficients = tuple(
                float(value) for value in data["distortion_coefficients"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed camera calibration") from error
        return cls(matrix, coefficients)

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraCalibration":
        try:
            with Path(path).open(encoding="utf-8") as file:
                return cls.from_dict(json.load(file))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load camera calibration: {path}") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_matrix": [list(row) for row in self.camera_matrix],
            "distortion_coefficients": list(self.distortion_coefficients),
        }


@dataclass(frozen=True)
class DetectedMarker:
    id: int
    corners: tuple[tuple[float, float], ...]
    center_x: float
    center_y: float
    area_px: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "corners": [
                [round(x, 2), round(y, 2)] for x, y in self.corners
            ],
            "center": [round(self.center_x, 2), round(self.center_y, 2)],
            "area_px": round(self.area_px, 2),
        }


@dataclass(frozen=True)
class MarkerPairGeometry:
    first_id: int
    second_id: int
    pixel_delta: tuple[float, float]
    physical_delta_cm: tuple[float, float]
    pixel_distance: float
    physical_distance_cm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_id": self.first_id,
            "second_id": self.second_id,
            "pixel_delta": [
                round(value, 2) for value in self.pixel_delta
            ],
            "physical_delta_cm": [
                round(value, 2) for value in self.physical_delta_cm
            ],
            "pixel_distance": round(self.pixel_distance, 2),
            "physical_distance_cm": round(self.physical_distance_cm, 2),
        }


@dataclass(frozen=True)
class HomographyResult:
    matrix: tuple[tuple[float, float, float], ...]
    reprojection_error_cm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": [
                [round(value, 8) for value in row] for row in self.matrix
            ],
            "reprojection_error_cm": round(self.reprojection_error_cm, 6),
        }


@dataclass(frozen=True)
class PersonEndpoints:
    box: tuple[int, int, int, int]
    top_of_head: tuple[float, float]
    bottom_of_feet: tuple[float, float]
    score: float

    def __post_init__(self) -> None:
        x, y, width, height = self.box
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(
                "person box must have a non-negative position and "
                "positive dimensions"
            )

        if not all(
            isfinite(value)
            for point in (self.top_of_head, self.bottom_of_feet)
            for value in point
        ):
            raise ValueError("person endpoints must have finite coordinates")

        if not isfinite(self.score):
            raise ValueError("person score must be finite")

        if self.top_of_head[1] >= self.bottom_of_feet[1]:
            raise ValueError("top of head must be above bottom of feet")

    @property
    def height_px(self) -> float:
        return self.bottom_of_feet[1] - self.top_of_head[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": list(self.box),
            "top_of_head": [round(value, 2) for value in self.top_of_head],
            "bottom_of_feet": [
                round(value, 2) for value in self.bottom_of_feet
            ],
            "height_px": round(self.height_px, 2),
            "score": round(self.score, 2),
        }


@dataclass(frozen=True)
class CalibrationResult:
    markers: tuple[DetectedMarker, ...]
    geometry: tuple[MarkerPairGeometry, ...]
    cm_per_pixel: float
    homography: HomographyResult
    person: PersonEndpoints | None = None
    perspective_height_cm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        person = None
        if self.person is not None:
            person = self.person.to_dict()
            person["height_cm"] = round(
                calculate_height_cm(self.person.height_px, self.cm_per_pixel),
                2,
            )
            if self.perspective_height_cm is not None:
                person["perspective_height_cm"] = round(
                    self.perspective_height_cm,
                    2,
                )

        return {
            "markers": [marker.to_dict() for marker in self.markers],
            "geometry": [pair.to_dict() for pair in self.geometry],
            "scale": {
                "cm_per_pixel": round(self.cm_per_pixel, 6),
                "pixels_per_cm": round(1 / self.cm_per_pixel, 2),
            },
            "homography": self.homography.to_dict(),
            "person": person,
        }


@dataclass(frozen=True)
class MarkerLayout:
    dictionary: str
    marker_size_cm: float
    markers: tuple[MarkerPosition, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarkerLayout":
        if not isinstance(data, dict):
            raise ValueError("marker layout must be a JSON object")

        try:
            marker_data = data["markers"]
            markers = tuple(
                MarkerPosition(
                    id=int(marker["id"]),
                    name=str(marker["name"]),
                    x_cm=float(marker["x_cm"]),
                    y_cm=float(marker["y_cm"]),
                )
                for marker in marker_data
            )
            layout = cls(
                dictionary=str(data["dictionary"]),
                marker_size_cm=float(data["marker_size_cm"]),
                markers=markers,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed marker layout configuration") from error
        layout.validate()
        return layout

    @classmethod
    def from_json(cls, path: str | Path) -> "MarkerLayout":
        with Path(path).open(encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    @property
    def marker_ids(self) -> tuple[int, ...]:
        return tuple(marker.id for marker in self.markers)

    def validate(self) -> None:
        if not isinstance(self.dictionary, str) or not self.dictionary.startswith("DICT_"):
            raise ValueError("dictionary must be an OpenCV ArUco dictionary name")
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown ArUco dictionary: {self.dictionary}")
        if not isfinite(self.marker_size_cm) or self.marker_size_cm <= 0:
            raise ValueError("marker_size_cm must be greater than zero")
        if len(self.markers) < 2:
            raise ValueError("at least two marker positions are required")
        if len(set(self.marker_ids)) != len(self.marker_ids):
            raise ValueError("marker IDs must be unique")
        names = [marker.name for marker in self.markers]
        if len(set(names)) != len(names):
            raise ValueError("marker names must be unique")
        positions = [(marker.x_cm, marker.y_cm) for marker in self.markers]
        if len(set(positions)) != len(positions):
            raise ValueError("marker physical positions must be unique")
