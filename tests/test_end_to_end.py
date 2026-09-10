import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).parents[1]


def make_layout(path: Path) -> None:
    layout = {
        "dictionary": "DICT_4X4_50",
        "marker_size_cm": 18,
        "markers": [
            {"id": 0, "name": "bottom_left", "x_cm": 0, "y_cm": 0},
            {"id": 1, "name": "bottom_right", "x_cm": 100, "y_cm": 0},
            {"id": 2, "name": "top_left", "x_cm": 0, "y_cm": 60},
            {"id": 3, "name": "top_right", "x_cm": 100, "y_cm": 60},
        ],
    }
    path.write_text(json.dumps(layout), encoding="utf-8")


def make_image(path: Path) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((500, 700), 255, dtype=np.uint8)
    for marker_id, (x, y) in enumerate(((50, 350), (550, 350), (50, 50), (550, 50))):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 100)
        image[y : y + 100, x : x + 100] = marker
    assert cv2.imwrite(str(path), image)


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "height_estimation", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_module_command_calibrates_image_and_writes_outputs(tmp_path):
    image_path = tmp_path / "markers.png"
    layout_path = tmp_path / "layout.json"
    result_path = tmp_path / "calibration.json"
    overlay_path = tmp_path / "calibration.png"
    make_image(image_path)
    make_layout(layout_path)

    completed = run_cli(
        str(image_path),
        "--layout",
        str(layout_path),
        "--output",
        str(result_path),
        "--overlay",
        str(overlay_path),
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert [marker["id"] for marker in result["markers"]] == [0, 1, 2, 3]
    assert len(result["geometry"]) == 6
    assert result["scale"]["cm_per_pixel"] == 0.2
    assert result["homography"]["reprojection_error_cm"] < 0.0001
    assert overlay_path.exists()
    assert "Wrote calibration overlay" in completed.stdout


def test_module_command_reports_missing_markers(tmp_path):
    image_path = tmp_path / "plain.png"
    layout_path = tmp_path / "layout.json"
    make_image(image_path)
    make_layout(layout_path)
    image = np.full((500, 700), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    completed = run_cli(str(image_path), "--layout", str(layout_path))

    assert completed.returncode != 0
    assert "missing required ArUco markers" in completed.stderr