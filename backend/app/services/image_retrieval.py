"""Match uploaded photos against approved reference photos (CLIP embeddings).

Reference library = rows of INDEX_CSV (Raw photos, test photos excluded) that have a
cached embedding. A reference photo is used only if its class has an approved record.
"""
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.services.embedder import embed_image_file
from app.store import store

BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
INDEX_CSV = BASE_DIR / "data" / "splits" / "main_index.csv"
EMBED_CACHE = BASE_DIR / "data" / "embeddings" / "ViT-B-32.npz"
CROP = "brinjal"
TOP_K = 5
# Classes that scored weakly in evaluation. Answers flag these as uncertain.
WEAK_CLASSES = {"Mosaic Virus", "Powdery Mildew"}


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


def _approved_record(label: str) -> dict | None:
    for rec in store.approved_records():
        if rec.get("crop", "").lower() == CROP and rec.get("condition", "").lower() == label.lower():
            return rec
    return None


def match_image(image_path: str, top_k: int = TOP_K) -> list[dict]:
    """Top-k reference photos for one uploaded image, restricted to approved classes."""
    feats, labels, paths = _reference_library()
    approved = {lab: rec for lab in set(labels) if (rec := _approved_record(lab))}
    if not approved:
        return []
    idx = np.array([i for i, lab in enumerate(labels) if lab in approved])
    q = embed_image_file(image_path)
    sims = feats[idx] @ q
    order = np.argsort(-sims)[:top_k]
    return [{
        "id": paths[idx[j]],
        "class": labels[idx[j]],
        "similarity": round(float(sims[j]), 3),
        "record": approved[labels[idx[j]]],
    } for j in order]


def evidence_from_matches(matches: list[dict]) -> list[dict]:
    return [{
        "id": m["id"],
        "type": "image",
        "title": f"{m['class']} reference photo",
        "source": m["record"].get("source_id"),
        "status": m["record"].get("review_status"),
        "similarity": m["similarity"],
    } for m in matches]


def candidates_from_matches(matches: list[dict]) -> list[dict]:
    by_class = defaultdict(list)
    for m in matches:
        by_class[m["class"]].append(m)
    total = len(matches)
    ranked = sorted(by_class.items(), key=lambda kv: (-len(kv[1]), -max(x["similarity"] for x in kv[1])))
    out = []
    for label, ms in ranked:
        mean_sim = sum(x["similarity"] for x in ms) / len(ms)
        against = ["Visual similarity only. Not confirmed by inspection or lab test."]
        if label in WEAK_CLASSES:
            against.append("This class matched weakly in evaluation. Treat this result as uncertain.")
        out.append({
            "name": label,
            "support": [f"{len(ms)} of {total} nearest reference photos", f"mean similarity {mean_sim:.2f}"],
            "against_or_missing": against,
            "evidence_ids": [x["id"] for x in ms],
        })
    return out