from __future__ import annotations

from pathlib import Path

from height_estimation.model_cache import download_model, get_model_cache_dir


HF_REPO_ID = "JunkyByte/easy_ViTPose"
HF_REPO_REVISION = "e83805274e89428969355ec4afffcbc413e79188"
DEFAULT_VITPOSE_MODEL = "l"
DEFAULT_YOLO_MODEL = "yolov8s"

VITPOSE_MODELS = {
    "s": "torch/wholebody/vitpose-s-wholebody.pth",
    "b": "torch/wholebody/vitpose-b-wholebody.pth",
    "l": "torch/wholebody/vitpose-l-wholebody.pth",
    "h": "torch/wholebody/vitpose-h-wholebody.pth",
}
YOLO_MODELS = {
    "yolov8s": "yolov8/yolov8s.pt",
    "yolov8m": "yolov8/yolov8m.pt",
}


def get_cache_dir() -> Path:
    return get_model_cache_dir()


def get_model_paths(
    vitpose_model: str = DEFAULT_VITPOSE_MODEL,
    yolo_model: str = DEFAULT_YOLO_MODEL,
) -> tuple[str, str]:
    if vitpose_model not in VITPOSE_MODELS:
        raise ValueError(
            f"Invalid ViTPose model {vitpose_model!r}; "
            f"available models: {', '.join(VITPOSE_MODELS)}"
        )
    if yolo_model not in YOLO_MODELS:
        raise ValueError(
            f"Invalid YOLO model {yolo_model!r}; "
            f"available models: {', '.join(YOLO_MODELS)}"
        )

    vitpose_filename = VITPOSE_MODELS[vitpose_model]
    yolo_filename = YOLO_MODELS[yolo_model]
    vitpose_path = download_model(
        HF_REPO_ID,
        vitpose_filename,
        revision=HF_REPO_REVISION,
        local_filename=vitpose_filename.rsplit("/", 1)[-1],
    )
    yolo_path = download_model(
        HF_REPO_ID,
        yolo_filename,
        revision=HF_REPO_REVISION,
        local_filename=yolo_filename.rsplit("/", 1)[-1],
    )
    return str(vitpose_path), str(yolo_path)
