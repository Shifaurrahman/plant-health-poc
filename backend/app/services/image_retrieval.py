"""Semantic search over the fused embedding cache.

Every query is embedded with embed_multimodal and compared with the cached vectors
by dot product (cosine, since vectors are unit length):
  photo only       -> {"image": p}
  photo + text     -> {"image": p, "text": t}
  text only        -> {"text": t}

The cache is EMBED_CACHE (e.g. manual_multimodal_embeddinggemma-2.npz).
No keyword matching is used.
"""
import csv
from collections import defaultdict
from functools import lru_cache

import numpy as np

from app.config import (
    CROP, DATA_DIR, EMBED_CACHE, INDEX_CSV, TEXT_CLASS_MARGIN,
    TEXT_ONLY_MIN_SIMILARITY, TOP_K,
)
from app.services.embedder import embed_multimodal
from app.store import store

TEXT_PHOTOS_PER_CLASS = 2  # text-only results: at most this many photos per class


def _local_photo_path(stored: str) -> str:
    """Map a Colab or absolute cache path onto the local data/raw tree when that file exists."""
    norm = stored.replace("\\", "/")
    key = "/data/raw/"
    i = norm.lower().find(key)
    if i >= 0:
        local = DATA_DIR / norm[i + len("/data/"):]
        if local.is_file():
            return str(local)
    return stored


@lru_cache(maxsize=1)
def _reference_library():
    if not EMBED_CACHE.exists():
        raise RuntimeError(f"embedding cache not found at {EMBED_CACHE}")
    data = np.load(EMBED_CACHE, allow_pickle=True)
    if "vectors" in data.files and "labels" in data.files:
        feats = np.asarray(data["vectors"], dtype=np.float32)
        labels = [str(x).replace("_", " ") for x in data["labels"].tolist()]
        paths = [_local_photo_path(str(x)) for x in data["paths"].tolist()]
        return feats, labels, paths

    if "feats" not in data.files:
        raise RuntimeError(
            f"{EMBED_CACHE.name} has keys {list(data.files)}; expected vectors+labels or feats."
        )
    if not INDEX_CSV.exists():
        raise RuntimeError(f"index CSV not found at {INDEX_CSV}")
    cached = dict(zip(data["paths"].tolist(), data["feats"]))
    with INDEX_CSV.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["phase"] == "Raw" and r["relative_path"] in cached]
    if not rows:
        raise RuntimeError(f"no Raw rows in {INDEX_CSV.name} match paths in {EMBED_CACHE.name}")
    feats = np.stack([cached[r["relative_path"]] for r in rows])
    labels = [r["class"].replace("_", " ") for r in rows]
    paths = [r["relative_path"] for r in rows]
    return feats, labels, paths


def _approved_by_label(crop: str) -> dict[str, dict]:
    """Approved records for one crop, keyed by lower-case condition name."""
    out: dict[str, dict] = {}
    for rec in store.approved_records():
        if (rec.get("crop") or "").lower() == crop:
            out.setdefault((rec.get("condition") or "").lower(), rec)
    return out


def _channel(image_path: str | None, text: str | None) -> str:
    if image_path and text and text.strip():
        return "image_text_fused"
    if image_path:
        return "image_only_fused"
    return "text_fused"


def _search(image_path: str | None, text: str | None, crop: str | None, top_k: int) -> list[dict]:
    crop = (crop or CROP).lower()
    feats, labels, paths = _reference_library()
    by_label = _approved_by_label(crop)
    link = {}
    for rec in store.approved_records():
        if (rec.get("crop") or "").lower() != crop:
            continue
        for img in rec.get("images") or []:
            if img.get("image_path"):
                link[img["image_path"]] = rec

    usable = [i for i, lab in enumerate(labels) if lab.lower() in by_label]
    if not usable:
        return []

    item = {}
    if image_path:
        item["image"] = image_path
    if text and text.strip():
        item["text"] = text.strip()
    if not item:
        raise ValueError("query needs an image, text, or both")

    q = embed_multimodal([item])[0]
    if q.shape[0] != feats.shape[1]:
        raise RuntimeError(
            f"embedding cache has {feats.shape[1]}-d vectors but the model gives {q.shape[0]}-d. "
            "Rebuild the cache with the same model, or fix PH_EMBED_BACKEND / PH_EMBED_CACHE_NAME."
        )

    idx = np.array(usable)
    sims = feats[idx] @ q
    order = np.argsort(-sims)[:top_k]
    channel = _channel(image_path, text)

    out = []
    for j in order:
        i = idx[j]
        rec = link.get(paths[i]) or by_label[labels[i].lower()]
        out.append({
            "id": paths[i],
            "class": labels[i],
            "similarity": round(float(sims[j]), 3),
            "record": rec,
            "channel": channel,
        })
    return out


def match_image(image_path: str, top_k: int = TOP_K, text: str | None = None,
                crop: str | None = None) -> list[dict]:
    """Photo (optionally with farmer text) searched against the fused cache."""
    return _search(image_path, text, crop, top_k)


def match_text(text: str, top_k: int = TOP_K, crop: str | None = None) -> list[dict]:
    """Farmer text only. Keeps only classes whose best score clears TEXT_ONLY_MIN_SIMILARITY
    and is within TEXT_CLASS_MARGIN of the top class, and at most TEXT_PHOTOS_PER_CLASS
    photos per class. Photos keep the cache order (highest score first)."""
    pool = _search(None, text, crop, top_k=max(top_k * 4, 40))
    if not pool:
        return []
    best_by_class: dict[str, float] = {}
    for m in pool:
        best_by_class[m["class"]] = max(best_by_class.get(m["class"], 0.0), m["similarity"])
    top_score = max(best_by_class.values())
    keep = {
        c for c, s in best_by_class.items()
        if s >= TEXT_ONLY_MIN_SIMILARITY and s >= top_score - TEXT_CLASS_MARGIN
    }
    out, per_class = [], {}
    for m in pool:
        if m["class"] not in keep:
            continue
        per_class[m["class"]] = per_class.get(m["class"], 0) + 1
        if per_class[m["class"]] <= TEXT_PHOTOS_PER_CLASS:
            out.append(m)
    return out[:top_k]


def evidence_from_matches(matches: list[dict]) -> list[dict]:
    return [{
        "id": m["id"],
        "type": "image",
        "title": f"{m['class']} reference photo",
        "condition": m["class"],
        "record_id": m["record"].get("record_id"),
        "source_id": m["record"].get("source_id"),
        "review_status": m["record"].get("review_status"),
        "reviewed_by": m["record"].get("reviewed_by"),
        "reviewed_on": m["record"].get("reviewed_on"),
        "similarity": m["similarity"],
        "channel": m.get("channel") or "image_only_fused",
        "label_basis": m["record"].get("label_basis"),
    } for m in matches]


def candidates_from_matches(matches: list[dict]) -> list[dict]:
    by_class = defaultdict(list)
    for m in matches:
        by_class[m["class"]].append(m)
    total = len(matches)
    ranked = sorted(by_class.items(), key=lambda kv: (-len(kv[1]), -max(x["similarity"] for x in kv[1])))
    out = []
    for label, ms in ranked:
        rec = ms[0]["record"]
        against = ["Visual similarity only. Not confirmed by inspection or lab test."]
        if rec.get("weak_in_evaluation"):
            against.append(rec.get("evaluation_note")
                           or "This class matched weakly in evaluation. Treat this result as uncertain.")
        out.append({
            "name": label,
            "support": [f"Matched {len(ms)} of {total} nearest reference photos"],
            "against_or_missing": against,
            "evidence_ids": [x["id"] for x in ms],
        })
    return out