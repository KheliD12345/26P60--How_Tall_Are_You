# Height Estimation

This project estimates human height from photographs using printed ArUco markers,
image geometry, body evidence, and uncertainty-aware measurement fusion.

## Quick Start

Use Python 3.12 or newer. Create a project-local environment and install the
core package:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On macOS or Linux, create and activate the environment with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Copy an image into the ignored `inputs/` directory, then run the measurement
pipeline:

```powershell
height-estimation --pipeline --quality-policy continue `
	--layout configs/marker_layout.json `
	--output outputs/person-height.json `
	inputs/person.jpg
```

The core install includes OpenCV's contrib modules for ArUco detection. The
default pipeline uses the built-in HOG/GrabCut person fallback. Optional model
packages must be installed separately; VGGHeads and ViTPose weights download
when their detectors are first used. Images in `inputs/` and generated files in
`outputs/` remain local and are not tracked.

## Calibration

Calibration-only mode remains available for marker inspection and is the default
when `--pipeline` is omitted:

```powershell
python -m height_estimation person.jpg `
	--layout configs/marker_layout.json `
	--output outputs/calibration.json `
	--overlay outputs/calibration.png
```

The calibration JSON contains detected markers, marker-pair geometry, the
centimetres-per-pixel scale, homography diagnostics, and optional camera
calibration metadata.

## Hybrid AprilTag Layout

The hybrid images contain four frame tags and two pole tags. The experimental
runner uses only pole tags 4 and 5, whose centre-to-centre distance is 81 cm:
that distance is stored as `stick_reference_distance_cm` in both marker layout
files. The AprilTag layout uses the same marker positions and size as
`configs/marker_layout.json`; only its dictionary differs.

```powershell
python experimental/apriltag_height_pipeline.py
```

The main pipeline uses the AprilTag layout for frame tags 0 to 3:

```powershell
python -m height_estimation --pipeline --batch `
	--quality-policy continue `
	--layout configs/apriltag_hybrid_layout.json `
	--output-dir outputs/apriltag-hybrid `
	inputs/ApriltagHybrid1.jpg inputs/ApriltagHybrid2.jpg
```

The main calibration stage ignores detected marker IDs that are not declared
by the selected layout, so pole tags 4 and 5 do not enter the frame homography.

## Full Pipeline

Run the complete single-image pipeline explicitly with `--pipeline`:

```powershell
python -m height_estimation --pipeline person.jpg `
	--layout configs/marker_layout.json `
	--quality-policy fail `
	--output outputs/person-height.json `
	--overlay outputs/person-overlay.png
```

The final JSON reports `geometric_height_cm` and acquisition quality. The
pipeline does not generate head-bounding-box estimates or fuse measurements.
The process exits with status `2` when a typed pipeline failure occurs. Use
`--quality-policy continue` to retain a result when the quality gate reports
warnings.

Only the direct geometric estimate is included in pipeline results.

Use `--no-person-fallback` to require configured model-backed body detectors.
This is useful for verifying detector availability without silently using the
OpenCV fallback.

## Model-Backed Detections

Install the model dependencies and test tools with:

```powershell
python -m pip install -e ".[models,test]"
```

The VGGHeads and ViTPose detector implementations are included in `src/`.
Their pinned weights are downloaded on first use into
`%USERPROFILE%\.cache\height-estimation`. Set `HEIGHT_ESTIMATION_MODEL_CACHE`
to use another shared location.

The model-only frame/stick runner below uses VGGHeads and ViTPose, disables the
HOG fallback, and reports whether both detectors succeeded:

```powershell
python experimental\apriltag_height_pipeline.py `
	--marker-type aruco `
	--reference-ids 4 5 `
	--frame-layout configs\marker_layout.json `
	--output-dir outputs\model-run `
	inputs\P001_1.jpg inputs\P002_1.jpg inputs\P003_1.jpg
```

The main pipeline can also request the model adapters explicitly:

```powershell
height-estimation --pipeline --model-detectors --no-person-fallback `
	--layout configs/marker_layout.json `
	--output outputs/model-result.json `
	inputs/P001_1.jpg
```

`--no-person-fallback` prevents HOG/GrabCut from substituting for unavailable
models. Optional MediaPipe segmentation and hand detectors are not required by
the frame/stick runner.

## Batch Processing

Batch mode accepts multiple image paths and writes one result per image. By
default, results are written below `outputs`; provide `--output-dir` to choose a
different directory:

```powershell
python -m height_estimation --pipeline --batch `
	--layout configs/marker_layout.json `
	--output-dir outputs/batch `
	--continue-on-error `
	person-one.jpg person-two.jpg
```

Without `--continue-on-error`, processing stops at the first failed image. With
it, all images are attempted and the command still returns status `2` if any
image failed.

## Excel Batch Interface

Place `Participant Heights.xlsx` in the repository root and participant photos
in `inputs/`. If the workbook is missing, the command creates it with the
expected headers and participant IDs from the selected images. Measured-height
cells remain blank in a new workbook. Run the model-only frame and stick
estimates with:

```powershell
height-estimation-excel
```

The command chooses each participant's `_2` image when present, otherwise `_1`.
It writes the frame estimate from markers 0–3 and the stick estimate from
markers 4 and 5 into the matching spreadsheet row. VGGHeads and ViTPose provide
the body endpoints; the HOG fallback is disabled. Each estimate runs
independently, so a failed method leaves its cell blank while other methods and
images continue. The measured-height column is preserved, and estimate columns
are refreshed on each run. Partial results are saved before the command returns
status `2` when any image or method fails.

## Validation

Install the test dependencies and run the checks from the repository root:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m compileall -q src tests
```