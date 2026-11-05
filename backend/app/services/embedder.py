"""Qwen3-VL-Embedding embedder. Text and images are mapped into ONE shared vector space.

Loaded once, on first use. Setup: clone https://github.com/QwenLM/Qwen3-VL-Embedding and set
PH_QWEN_REPO_DIR to that folder.

Every function returns unit-length float32 numpy vectors, so a dot product is cosine similarity.
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from app.config import (
    EMBED_BATCH, EMBED_DTYPE, EMBED_MAX_PIXELS, EMBED_MODEL, EMBED_QUERY_INSTRUCTION, QWEN_REPO_DIR,
)

# Re-exported so versions.py and scripts can import a single name.
MODEL = EMBED_MODEL

# "manual" reads the weights as bytes (works on this Windows CPU build).
# "hf" uses the repo's normal from_pretrained path, which crashes here.
LOADER = os.getenv("PH_EMBED_LOADER", "manual").lower()


def _manual_from_pretrained(cls, model_name_or_path, *args, **kwargs):
    """Replacement for Qwen3VLForEmbedding.from_pretrained that avoids memory-mapped safetensors."""
    import safetensors.torch as st
    from huggingface_hub import hf_hub_download
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    dtype = kwargs.get("dtype") or kwargs.get("torch_dtype") or torch.bfloat16
    config = Qwen3VLConfig.from_pretrained(model_name_or_path)

    if os.path.isdir(model_name_or_path):
        weights = os.path.join(model_name_or_path, "model.safetensors")
    else:
        weights = hf_hub_download(model_name_or_path, "model.safetensors")

    with open(weights, "rb") as f:
        raw = f.read()
    state = st.load(raw)
    del raw

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = cls(config)
    finally:
        torch.set_default_dtype(previous)

    result = model.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys:
        raise RuntimeError(
            f"{len(result.missing_keys)} model weights not found in checkpoint, "
            f"e.g. {result.missing_keys[:3]}"
        )
    if result.unexpected_keys:
        print(f"note: {len(result.unexpected_keys)} checkpoint keys unused, e.g. {result.unexpected_keys[:3]}")
    return model.eval()


@lru_cache(maxsize=1)
def _load():
    if QWEN_REPO_DIR and QWEN_REPO_DIR not in sys.path:
        sys.path.insert(0, QWEN_REPO_DIR)
    try:
        import src.models.qwen3_vl_embedding as qwen_mod
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Qwen3VLEmbedder. Check PH_QWEN_REPO_DIR in .env points to the "
            "Qwen3-VL-Embedding folder that contains src/."
        ) from e

    if LOADER == "manual":
        qwen_mod.Qwen3VLForEmbedding.from_pretrained = classmethod(_manual_from_pretrained)

    kwargs = {"dtype": getattr(torch, EMBED_DTYPE)}
    if EMBED_MAX_PIXELS:
        kwargs["max_pixels"] = EMBED_MAX_PIXELS
    return qwen_mod.Qwen3VLEmbedder(model_name_or_path=EMBED_MODEL, **kwargs)


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