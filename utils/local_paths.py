"""Utilities to force all model caches/artifacts into the project folder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def ensure_local_model_env(base_dir: str = "models") -> Dict[str, Path]:
    """Set environment variables so model downloads stay inside repository.

    Returns a dictionary of created local directories.
    """
    root = Path(base_dir)
    hf_home = root / "huggingface"
    st_cache = root / "sentence_transformers"
    whisper_cache = root / "whisper"
    paddle_home = root / "paddle"

    dirs = {
        "root": root,
        "hf_home": hf_home,
        "sentence_transformers": st_cache,
        "whisper": whisper_cache,
        "paddle": paddle_home,
    }

    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home.resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(st_cache.resolve())
    os.environ["PADDLE_HOME"] = str(paddle_home.resolve())

    return dirs
