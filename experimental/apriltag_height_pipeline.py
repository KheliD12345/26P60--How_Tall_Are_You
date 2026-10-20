"""Compare frame-tag rectification with the independent stick-tag scale.

The experimental runner uses the same ViTPose heel and VGGHeads head-top
landmarks as the production pipeline. Four frame markers define the primary
homography when available. A separate scale from stick markers 4 and 5 is
reported as a calibration check.
"""

from __future__ import annotations

from height_estimation.body_detection import (
    BodyDetectionConfig,
    BodyDetectionOrchestrator,
    ViTPoseDetectorAdapter,
    VGGHeadsDetectorAdapter,
)
from height_estimation.frame_stick import (
    DEFAULT_MARKER_TYPE,
    MARKER_CONFIGS,
    process_image,
)
from height_estimation.frame_stick_cli import (
    FRAME_LAYOUT_PATH,
    IMAGE_NAMES,
    SCRIPT_DIR,
    _resolve_path,
    main,
    parse_args,
)
from height_estimation.models import MarkerLayout


if __name__ == "__main__":
    raise SystemExit(main())
