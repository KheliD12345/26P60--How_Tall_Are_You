# Height Estimation

Estimate height from photos using ArUco frame markers, stick markers, VGGHeads,
and ViTPose.

## Quick start

Use Python 3.12 or newer. In PowerShell, install the project and model support:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[models]"
```

Put supported photos in `inputs/`, then run either batch command:

```powershell
height-estimation-json
height-estimation-excel
```

Both commands use the ArUco layout in `configs/marker_layout.json`, check frame
markers 0–3 and stick markers 4–5, and use VGGHeads and ViTPose for body
endpoints. The HOG fallback is disabled. Each estimate is attempted
independently; a missing marker set leaves only that method's result blank.
Partial outputs are saved, and the command returns status `2` if any image or
method fails.

`height-estimation-json` writes one file per participant to `outputs/`, with
`geometric_height_cm` retained as the frame-first primary result (stick fallback)
and separate `frame_estimated_height_cm` and `stick_estimated_height_cm` fields.
`height-estimation-excel` creates or updates
`outputs/Participant Heights.xlsx`; it preserves measured heights and refreshes
the frame and stick estimate columns. Both accept `--input-dir`; the JSON
command also accepts `--output-dir`, and the Excel command accepts `--workbook`.

Images are processed in case-insensitive filename order. A filename beginning
with `P<number>` keeps that participant number. Other names receive the next
unused ID, starting at `P001`; duplicate participant IDs are rejected. Model
weights download on first use to `%USERPROFILE%\.cache\height-estimation`.
Set `HEIGHT_ESTIMATION_MODEL_CACHE` to use another cache directory.

## Legacy CLI

`height-estimation` remains available for calibration and the geometric-only
pipeline. Without `--pipeline`, it inspects markers; with `--pipeline`, it runs
the legacy full pipeline. For calibration-only inspection:

```powershell
height-estimation inputs/person.jpg `
    --layout configs/marker_layout.json `
    --output outputs/calibration.json `
    --overlay outputs/calibration.png
```

For one geometric height result:

```powershell
height-estimation --pipeline inputs/person.jpg `
    --layout configs/marker_layout.json `
    --quality-policy continue `
    --output outputs/person-height.json
```

The legacy batch form accepts explicit image paths:

```powershell
height-estimation --pipeline --batch --continue-on-error `
    --layout configs/marker_layout.json `
    --output-dir outputs/legacy `
    inputs/one.jpg inputs/two.jpg
```

Command options:

- `--layout`: marker layout JSON; defaults to `configs/marker_layout.json`.
- `--version`: print the installed package version.
- `--pipeline`: run height estimation instead of calibration-only inspection.
- `--batch`: allow multiple positional images; requires `--pipeline`.
- `--output` / `--output-dir`: choose a single result file or batch directory.
- `--continue-on-error`: attempt remaining batch images after a failure.
- `--quality-policy fail|continue`: stop or continue when quality checks fail.
- `--camera-calibration`: provide camera calibration for undistortion.
- `--overlay`: write a calibration or pipeline overlay image.
- `--model-detectors`: opt into optional model detectors in the legacy pipeline.
- `--no-person-fallback`: disable its HOG/GrabCut fallback.

The experimental `experimental/apriltag_height_pipeline.py` runner accepts
explicit image paths. Its `--marker-type`, `--reference-ids`, `--frame-layout`,
and `--output-dir` options configure AprilTag/ArUco comparisons.

## Validation

Install test dependencies and run from the repository root:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m compileall -q src tests
```
