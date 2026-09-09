from pathlib import Path

import pytest

from height_mvp.models import MarkerLayout


LAYOUT_PATH = Path(__file__).parents[1] / "configs" / "marker_layout.json"


def test_loads_backdrop_layout():
    layout = MarkerLayout.from_json(LAYOUT_PATH)

    assert layout.dictionary == "DICT_4X4_50"
    assert layout.marker_size_cm == 18.0
    assert layout.marker_ids == (0, 1, 2, 3)


def test_rejects_duplicate_marker_ids():
    data = {
        "dictionary": "DICT_4X4_50",
        "marker_size_cm": 18,
        "markers": [
            {"id": 0, "name": "left", "x_cm": 0, "y_cm": 0},
            {"id": 0, "name": "right", "x_cm": 90, "y_cm": 0},
        ],
    }

    with pytest.raises(ValueError, match="unique"):
        MarkerLayout.from_dict(data)
