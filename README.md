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

## Optional Model Adapters

Install the model dependencies and test tools with:

```powershell
python -m pip install -e ".[models,test]"
```

The adapters also require the external detector source modules containing
`vitpose_detection`, `vggheads_detection`, and `mediapipe_detection` to be
available through the normal Python environment. The application does not
modify `sys.path` at runtime.

Optional model adapters can be enabled from the CLI. They are constructed
lazily on first inference call:

```powershell
python -m height_estimation --pipeline --model-detectors `
	--quality-policy continue `
	person.jpg --layout configs/marker_layout.json
```

They can also be wired explicitly through `PipelineConfig` when external
detector packages are available.

```python
from height_estimation.body_detection import (
	BodyDetectionConfig,
	MediaPipeHandDetectorAdapter,
	MediaPipeSegmentationAdapter,
	VGGHeadsDetectorAdapter,
	ViTPoseDetectorAdapter,
)
from height_estimation.pipeline import PipelineConfig

config = PipelineConfig(
	use_person_fallback=True,
	body_detection=BodyDetectionConfig(
		pose_detector=ViTPoseDetectorAdapter(),
		head_detector=VGGHeadsDetectorAdapter(),
		segmentation_detector=MediaPipeSegmentationAdapter(),
		hand_detector=MediaPipeHandDetectorAdapter(),
	),
)
```

When an optional dependency or model file is unavailable, the corresponding
detector returns an unavailable status and the pipeline continues to use
available evidence. Runtime detector errors remain failure statuses and are
included in diagnostics.

Model weights are downloaded by the external detector implementations when
first used. Configure model paths and network access before running a
model-backed pipeline.

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