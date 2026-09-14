from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarkerPosition:
    id: int
    name: str
    x_cm: float
    y_cm: float


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

    def to_dict(self) -> dict[str, Any]:
        person = None
        if self.person is not None:
            person = self.person.to_dict()
            person["height_cm"] = round(
                self.person.height_px * self.cm_per_pixel, 2
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
        markers = tuple(
            MarkerPosition(
                id=int(marker["id"]),
                name=str(marker["name"]),
                x_cm=float(marker["x_cm"]),
                y_cm=float(marker["y_cm"]),
            )
            for marker in data["markers"]
        )
        layout = cls(
            dictionary=str(data["dictionary"]),
            marker_size_cm=float(data["marker_size_cm"]),
            markers=markers,
        )
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
        if not self.dictionary.startswith("DICT_"):
            raise ValueError("dictionary must be an OpenCV ArUco dictionary name")
        if self.marker_size_cm <= 0:
            raise ValueError("marker_size_cm must be greater than zero")
        if len(self.markers) < 2:
            raise ValueError("at least two marker positions are required")
        if len(set(self.marker_ids)) != len(self.marker_ids):
            raise ValueError("marker IDs must be unique")
