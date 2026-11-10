"""Embedder. Backend is chosen by PH_EMBED_BACKEND: "qwen" (default) or "gemma2".

Every function returns unit-length float32 numpy vectors, so dot product = cosine similarity.
"""
import sys
from functools import lru_cache

import numpy as np
import torch

from app.config import (
    EMBED_BACKEND, EMBED_BATCH, EMBED_DTYPE, EMBED_MODEL,
    EMBED_QUERY_INSTRUCTION, GEMMA2_DTYPE, GEMMA2_MODEL, QWEN_REPO_DIR,
)

MODEL = GEMMA2_MODEL if EMBED_BACKEND == "gemma2" else EMBED_MODEL

# Task prefixes from the EmbeddingGemma model card. Images get no prefix.
GEMMA2_QUERY_PREFIX = "task: search result | query: "
GEMMA2_DOC_PREFIX = "title: none | text: "


def _unit(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# ---------------- EmbeddingGemma ----------------
@lru_cache(maxsize=1)
def _load_gemma2():
    from sentence_transformers import SentenceTransformer
    dtype = getattr(torch, GEMMA2_DTYPE)
    if dtype == torch.float16:
        raise ValueError("EmbeddingGemma must not run in float16.")
    return SentenceTransformer(GEMMA2_MODEL, model_kwargs={"torch_dtype": dtype})


def _gemma2_encode(inputs) -> np.ndarray:
    model = _load_gemma2()
    with torch.no_grad():
        out = model.encode(inputs, batch_size=EMBED_BATCH,
                           normalize_embeddings=True, convert_to_numpy=True)
    return _unit(out)


# ---------------- Qwen3-VL (unchanged) ----------------
@lru_cache(maxsize=1)
def _load_qwen():
    if QWEN_REPO_DIR and QWEN_REPO_DIR not in sys.path:
        sys.path.insert(0, QWEN_REPO_DIR)
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    return Qwen3VLEmbedder(model_name_or_path=MODEL, dtype=getattr(torch, EMBED_DTYPE))


def _qwen_embed(inputs: list[dict], batch: int = EMBED_BATCH) -> np.ndarray:
    model = _load_qwen()
    parts = [_unit(model.process(inputs[i:i + batch])) for i in range(0, len(inputs), batch)]
    return np.concatenate(parts)


# ---------------- public API ----------------
def embed_image_file(path) -> np.ndarray:
    if EMBED_BACKEND == "gemma2":
        return _gemma2_encode([{"image": str(path)}])[0]
    return _qwen_embed([{"image": str(path)}])[0]


def embed_image_files(paths, batch: int = EMBED_BATCH) -> np.ndarray:
    if EMBED_BACKEND == "gemma2":
        return _gemma2_encode([{"image": str(p)} for p in paths])
    return _qwen_embed([{"image": str(p)} for p in paths], batch)


def encode_document(texts: list[str]) -> np.ndarray:
    if EMBED_BACKEND == "gemma2":
        return _gemma2_encode([GEMMA2_DOC_PREFIX + t for t in texts])
    return _qwen_embed([{"text": t} for t in texts])


def encode_query(texts: list[str]) -> np.ndarray:
    if EMBED_BACKEND == "gemma2":
        return _gemma2_encode([GEMMA2_QUERY_PREFIX + t for t in texts])
    return _qwen_embed([{"text": t, "instruction": EMBED_QUERY_INSTRUCTION} for t in texts])