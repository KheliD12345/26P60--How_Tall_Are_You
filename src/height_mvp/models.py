from dataclasses import dataclass
import json
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
            "corners": [[round(x, 2), round(y, 2)] for x, y in self.corners],
            "center": [round(self.center_x, 2), round(self.center_y, 2)],
            "area_px": round(self.area_px, 2),
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
