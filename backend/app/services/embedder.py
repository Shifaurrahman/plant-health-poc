"""Qwen3-VL-Embedding embedder. Text and images are mapped into ONE shared vector space.

Loaded once, on first use. Setup: clone https://github.com/QwenLM/Qwen3-VL-Embedding and set
PH_QWEN_REPO_DIR to that folder (the Qwen3VLEmbedder class is in the repo, not on pip).

Every function returns unit-length float32 numpy vectors, so a dot product is cosine similarity.
Query vs document: Qwen3-VL-Embedding is instruction-aware. Questions are encoded with an
instruction (encode_query); record text and reference photos are encoded without one.
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import (
    EMBED_BATCH, EMBED_DTYPE, EMBED_MAX_PIXELS, EMBED_MODEL, EMBED_QUERY_INSTRUCTION, QWEN_REPO_DIR,
)

# Re-exported so versions.py and scripts can import a single name.
MODEL = EMBED_MODEL


@lru_cache(maxsize=1)
def _load():
    import torch

    if QWEN_REPO_DIR and QWEN_REPO_DIR not in sys.path:
        sys.path.insert(0, QWEN_REPO_DIR)
    try:
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Qwen3VLEmbedder. Clone QwenLM/Qwen3-VL-Embedding, install its "
            "requirements, and set PH_QWEN_REPO_DIR to that folder."
        ) from e

    kwargs = {"model_name_or_path": EMBED_MODEL, "dtype": getattr(torch, EMBED_DTYPE), "low_cpu_mem_usage": False}
    if EMBED_MAX_PIXELS:
        kwargs["max_pixels"] = EMBED_MAX_PIXELS
    return Qwen3VLEmbedder(**kwargs)


def _unit(x) -> np.ndarray:
    """Tensor or array -> L2-normalised float32 rows."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _embed(inputs: list[dict], batch: int = EMBED_BATCH) -> np.ndarray:
    if not inputs:
        return np.zeros((0, 0), dtype=np.float32)
    model = _load()
    parts = [_unit(model.process(inputs[i:i + batch])) for i in range(0, len(inputs), batch)]
    return np.concatenate(parts)


def embed_image_file(path: str | Path) -> np.ndarray:
    """One photo -> one vector. Used for uploads at request time."""
    return _embed([{"image": str(path)}])[0]


def embed_image_files(paths: list, batch: int = EMBED_BATCH) -> np.ndarray:
    """Many photos -> one row each. Used when building the reference cache offline."""
    return _embed([{"image": str(p)} for p in paths], batch)


def encode_document(texts: list[str]) -> np.ndarray:
    """Text being searched (expert record text). No instruction."""
    return _embed([{"text": t} for t in texts])


def encode_query(texts: list[str]) -> np.ndarray:
    """Text doing the searching (the farmer's question). Carries the retrieval instruction."""
    return _embed([{"text": t, "instruction": EMBED_QUERY_INSTRUCTION} for t in texts])