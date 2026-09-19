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

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("marker IDs must be non-negative")
        if len(self.corners) != 4 or any(len(point) != 2 for point in self.corners):
            raise ValueError("detected markers must have four corner points")
        if not all(
            isfinite(value)
            for point in self.corners
            for value in point
        ):
            raise ValueError("marker coordinates must be finite")
        if not all(isfinite(value) for value in (self.center_x, self.center_y)):
            raise ValueError("marker centres must be finite")
        if not isfinite(self.area_px) or self.area_px <= 0:
            raise ValueError("marker area must be finite and greater than zero")

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

    def __post_init__(self) -> None:
        if self.first_id < 0 or self.second_id < 0:
            raise ValueError("marker IDs must be non-negative")
        if self.first_id == self.second_id:
            raise ValueError("marker geometry requires two distinct IDs")
        if len(self.pixel_delta) != 2 or len(self.physical_delta_cm) != 2:
            raise ValueError("marker deltas must contain two values")
        if not all(
            isfinite(value)
            for value in (*self.pixel_delta, *self.physical_delta_cm)
        ):
            raise ValueError("marker geometry values must be finite")
        if not isfinite(self.pixel_distance) or self.pixel_distance <= 0:
            raise ValueError("pixel distance must be finite and greater than zero")
        if (
            not isfinite(self.physical_distance_cm)
            or self.physical_distance_cm <= 0
        ):
            raise ValueError(
                "physical distance must be finite and greater than zero"
            )

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

    def __post_init__(self) -> None:
        if len(self.matrix) != 3 or any(len(row) != 3 for row in self.matrix):
            raise ValueError("homography matrix must have shape 3x3")
        if not all(isfinite(value) for row in self.matrix for value in row):
            raise ValueError("homography matrix must contain finite values")
        if not isfinite(self.reprojection_error_cm) or self.reprojection_error_cm < 0:
            raise ValueError("homography reprojection error must be non-negative")

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
    camera_calibration: CameraCalibration | None = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.cm_per_pixel) or self.cm_per_pixel <= 0:
            raise ValueError("cm_per_pixel must be finite and greater than zero")
        if not all(isinstance(item, str) and item for item in self.diagnostics):
            raise ValueError("calibration diagnostics must be non-empty strings")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationResult":
        if not isinstance(data, dict):
            raise ValueError("calibration result must be a JSON object")
        try:
            markers = tuple(
                DetectedMarker(
                    id=int(marker["id"]),
                    corners=tuple(
                        (float(point[0]), float(point[1]))
                        for point in marker["corners"]
                    ),
                    center_x=float(marker["center"][0]),
                    center_y=float(marker["center"][1]),
                    area_px=float(marker["area_px"]),
                )
                for marker in data["markers"]
            )
            geometry = tuple(
                MarkerPairGeometry(
                    first_id=int(pair["first_id"]),
                    second_id=int(pair["second_id"]),
                    pixel_delta=tuple(float(value) for value in pair["pixel_delta"]),
                    physical_delta_cm=tuple(
                        float(value) for value in pair["physical_delta_cm"]
                    ),
                    pixel_distance=float(pair["pixel_distance"]),
                    physical_distance_cm=float(pair["physical_distance_cm"]),
                )
                for pair in data["geometry"]
            )
            scale = data["scale"]
            homography_data = data["homography"]
            homography = HomographyResult(
                matrix=tuple(
                    tuple(float(value) for value in row)
                    for row in homography_data["matrix"]
                ),
                reprojection_error_cm=float(
                    homography_data["reprojection_error_cm"]
                ),
            )
            person_data = data.get("person")
            person = None
            perspective_height_cm = None
            if person_data is not None:
                person = PersonEndpoints(
                    box=tuple(int(value) for value in person_data["box"]),
                    top_of_head=tuple(
                        float(value) for value in person_data["top_of_head"]
                    ),
                    bottom_of_feet=tuple(
                        float(value) for value in person_data["bottom_of_feet"]
                    ),
                    score=float(person_data["score"]),
                )
                if "perspective_height_cm" in person_data:
                    perspective_height_cm = float(
                        person_data["perspective_height_cm"]
                    )
            camera_data = data.get("camera_calibration")
            return cls(
                markers=markers,
                geometry=geometry,
                cm_per_pixel=float(scale["cm_per_pixel"]),
                homography=homography,
                person=person,
                perspective_height_cm=perspective_height_cm,
                camera_calibration=(
                    None
                    if camera_data is None
                    else CameraCalibration.from_dict(camera_data)
                ),
                diagnostics=tuple(data.get("diagnostics", ())),
            )
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError("malformed calibration result") from error

    @classmethod
    def from_json(cls, path: str | Path) -> "CalibrationResult":
        try:
            with Path(path).open(encoding="utf-8") as file:
                return cls.from_dict(json.load(file))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load calibration result: {path}") from error

    def to_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

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

        result = {
            "markers": [marker.to_dict() for marker in self.markers],
            "geometry": [pair.to_dict() for pair in self.geometry],
            "scale": {
                "cm_per_pixel": round(self.cm_per_pixel, 6),
                "pixels_per_cm": round(1 / self.cm_per_pixel, 2),
            },
            "homography": self.homography.to_dict(),
            "person": person,
        }
        if self.camera_calibration is not None:
            result["camera_calibration"] = self.camera_calibration.to_dict()
        if self.diagnostics:
            result["diagnostics"] = list(self.diagnostics)
        return result


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
