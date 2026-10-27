"""CLIP image embedder. Loaded once, on first use.

Same model and preprocessing as scripts/evaluate_retrieval.py, so query embeddings
are comparable with the cached reference embeddings.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


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