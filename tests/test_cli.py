import json

import pytest

from height_mvp.cli import main
from height_mvp.models import DetectedMarker


def test_cli_loads_default_layout(capsys):
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "Loaded 4 markers" in output
    assert "DICT_4X4_50" in output


def test_cli_writes_detection_json(tmp_path, monkeypatch, capsys):
    markers = tuple(
        DetectedMarker(
            id=marker_id,
            corners=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)),
            center_x=2.0,
            center_y=3.0,
            area_px=4.0,
        )
        for marker_id in range(4)
    )
    monkeypatch.setattr("height_mvp.cli.detect_markers", lambda image, layout: markers)
    image_path = tmp_path / "image.jpg"
    output_path = tmp_path / "detections.json"
    image_path.write_bytes(b"image")

    assert main([str(image_path), "--output", str(output_path)]) == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["markers"][0]["id"] == 0
    assert "Wrote 4 markers" in capsys.readouterr().out


def test_cli_rejects_missing_markers(tmp_path, monkeypatch, capsys):
    marker = DetectedMarker(
        id=0,
        corners=((1.0, 2.0),) * 4,
        center_x=1.0,
        center_y=2.0,
        area_px=1.0,
    )
    monkeypatch.setattr("height_mvp.cli.detect_markers", lambda image, layout: (marker,))
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")

    with pytest.raises(SystemExit) as error:
        main([str(image_path)])

    assert error.value.code == 2
    assert "missing required ArUco markers" in capsys.readouterr().err
