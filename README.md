# Height Estimation

This project estimates human height from photographs using printed ArUco markers,
image geometry, body evidence, and uncertainty-aware measurement fusion.

## Setup

Use Python 3.12 or newer. Install the package and its test dependencies from the
project directory:

```powershell
python -m pip install -e ".[test]"
```

The package uses OpenCV's contrib modules for ArUco marker detection. Optional
model-backed detectors are not downloaded or imported automatically. The default
pipeline can use the built-in OpenCV HOG/GrabCut person fallback.

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

## Full Pipeline

Run the complete single-image pipeline explicitly with `--pipeline`:

```powershell
python -m height_estimation --pipeline person.jpg `
	--layout configs/marker_layout.json `
	--quality-policy fail `
	--confidence-level 0.95 `
	--output outputs/person-measurement.json `
	--overlay outputs/person-overlay.png
```

The final JSON uses the `MeasurementResult` contract and includes the fused
height estimate, uncertainty interval, quality assessment, method estimates,
fusion weights, diagnostics, and warnings. The process exits with status `2`
when a typed pipeline failure occurs. Use `--quality-policy continue` to retain
a result when the quality gate reports warnings.

Use `--no-person-fallback` to require configured model-backed body detectors.
This is useful for verifying detector availability without silently using the
OpenCV fallback.

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

## Validation

Run the complete test suite from this directory:

```powershell
python -m pytest -q
python -m compileall -q src tests

```