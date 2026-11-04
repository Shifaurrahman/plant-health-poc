"""Match uploaded photos against approved reference photos (Qwen3-VL-Embedding vectors).

Reference library = rows of INDEX_CSV (Raw photos, test photos excluded) that have a cached
embedding. A reference photo is used only if its class has an approved record.
A photo listed in a record's "images" field is linked to that record directly.
"""
import csv
from collections import defaultdict
from functools import lru_cache

import numpy as np

from app.config import CROP, EMBED_CACHE, INDEX_CSV, TOP_K
from app.services.embedder import embed_image_file
from app.store import store


@lru_cache(maxsize=1)
def _reference_library():
    if not EMBED_CACHE.exists():
        raise RuntimeError(f"embedding cache not found at {EMBED_CACHE}")
    if not INDEX_CSV.exists():
        raise RuntimeError(f"index CSV not found at {INDEX_CSV}")
    data = np.load(EMBED_CACHE, allow_pickle=True)
    cached = dict(zip(data["paths"].tolist(), data["feats"]))
    with INDEX_CSV.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["phase"] == "Raw" and r["relative_path"] in cached]
    feats = np.stack([cached[r["relative_path"]] for r in rows])
    labels = [r["class"].replace("_", " ") for r in rows]
    paths = [r["relative_path"] for r in rows]
    return feats, labels, paths


def _approved_by_label() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rec in store.approved_records():
        if rec.get("crop", "").lower() == CROP:
            out.setdefault(rec.get("condition", "").lower(), rec)
    return out


def match_image(image_path: str, top_k: int = TOP_K) -> list[dict]:
    """Top-k reference photos for one uploaded image, restricted to approved classes."""
    feats, labels, paths = _reference_library()
    by_label = _approved_by_label()
    link = {}
    for rec in store.approved_records():
        for img in rec.get("images") or []:
            if img.get("image_path"):
                link[img["image_path"]] = rec

    usable = [i for i, lab in enumerate(labels) if lab.lower() in by_label]
    if not usable:
        return []
    idx = np.array(usable)
    q = embed_image_file(image_path)
    if q.shape[0] != feats.shape[1]:
        raise RuntimeError(
            f"embedding cache has {feats.shape[1]}-d vectors but the model gives {q.shape[0]}-d. "
            "Rebuild the cache with scripts/evaluate_retrieval.py (or fix PH_EMBED_MODEL / PH_EMBED_CACHE_NAME).")
    sims = feats[idx] @ q
    order = np.argsort(-sims)[:top_k]
    out = []
    for j in order:
        i = idx[j]
        rec = link.get(paths[i]) or by_label[labels[i].lower()]
        out.append({
            "id": paths[i],
            "class": labels[i],
            "similarity": round(float(sims[j]), 3),
            "record": rec,
            "channel": "image_to_image",
        })
    return out


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
        "channel": m.get("channel") or "image_to_image",
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