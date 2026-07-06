"""Model weight download from Hugging Face Hub (SRS FR-2.8, §8.1).

Weights are cached under ``~/.cache/privyscope/`` and reused on subsequent calls.
``revision`` pins a version; ``HTTPS_PROXY`` is honoured by ``huggingface_hub``.
No network calls occur once a local ``cache_dir`` is supplied.
"""
from __future__ import annotations

import os
from pathlib import Path

# The weights repo is language-specific and supplied by the active language
# plugin (``LanguagePlugin.default_repo``); the core hardcodes nothing.
DEFAULT_CACHE = Path(os.path.expanduser("~/.cache/privyscope"))


def resolve_weights(
    repo_id: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a local directory containing the engine bundle.

    If ``cache_dir`` already contains ``privyscope_meta.json`` it is used directly
    (fully offline). Otherwise weights are downloaded from ``repo_id`` on the Hub.
    """
    if cache_dir is not None:
        local = Path(cache_dir)
        if (local / "privyscope_meta.json").exists():
            return local

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "downloading weights requires huggingface_hub: pip install huggingface_hub"
        ) from exc

    target = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(target),
        allow_patterns=["*.onnx", "*.json", "*.txt", "tokenizer*", "vocab*", "*.model", "spm*"],
    )
    return Path(path)
