from __future__ import annotations

import os
from pathlib import Path


def get_model_cache_dir() -> Path:
    configured = (
        os.environ.get("HEIGHT_ESTIMATION_MODEL_CACHE")
        or os.environ.get("VITPOSE_CACHE")
    )
    cache_dir = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "height-estimation"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_model(
    repo_id: str,
    filename: str,
    *,
    revision: str,
    local_filename: str | None = None,
) -> Path:
    cache_dir = get_model_cache_dir()
    if local_filename is not None:
        cached_path = cache_dir / local_filename
        if cached_path.is_file():
            return cached_path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(cache_dir),
        )
    )
