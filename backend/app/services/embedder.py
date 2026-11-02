"""CLIP image embedder. Loaded once, on first use.

Model name and weights come from config.py (PH_CLIP_MODEL / PH_CLIP_PRETRAINED).
Same settings must match the embedding cache used by image_retrieval.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import CLIP_MODEL, CLIP_PRETRAINED

# Re-exported so versions.py and scripts can import a single name.
MODEL = CLIP_MODEL
PRETRAINED = CLIP_PRETRAINED


@lru_cache(maxsize=1)
def _load():
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    return model, preprocess, torch


def embed_image_file(path: str | Path) -> np.ndarray:
    from PIL import Image

    model, preprocess, torch = _load()
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        f = model.encode_image(img)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]


def encode_text(texts: list[str]) -> np.ndarray:
    """CLIP text encoder, same model and embedding space as the images. Unit-length rows."""
    import open_clip

    model, _, torch = _load()
    tokens = open_clip.get_tokenizer(MODEL)(texts)  # long text is truncated to the 77-token context
    with torch.no_grad():
        t = model.encode_text(tokens)
        t = t / t.norm(dim=-1, keepdim=True)
    return t.cpu().numpy()
