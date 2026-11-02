"""Text and image-to-text retrieval over approved records.

Channels (plan §6):
  keyword        word overlap
  text_semantic  CLIP text query vs record text
  image_to_text  CLIP image query vs record text (shared embedding space)
"""
import re
from collections import defaultdict

import numpy as np

from app.services.embedder import embed_image_file, encode_text
from app.store import store

# Generic words that appear in almost every record or plant_part field.
# Without these, every record matches on "leaf" alone.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "on", "in", "is", "are", "with", "my", "this",
    "that", "it", "to", "for", "some", "have", "has", "lower", "upper", "leaves", "leaf",
    "spot", "spots", "plant", "brown", "about", "weeks", "week", "days", "day",
}
TOKEN = re.compile(r"[a-z]+")
RRF_K = 60

_text_vectors: dict[str, object] = {}  # record text -> embedding (computed once per text)


def _tokens(s: str) -> set[str]:
    return {t for t in TOKEN.findall(s.lower()) if t not in STOPWORDS}


def _record_text(rec: dict) -> str:
    return " ".join(x for x in [rec.get("condition", ""), rec.get("symptoms", ""),
                                rec.get("distinguishing_features", "")] if x)


def _text_vector(text: str):
    if text not in _text_vectors:
        _text_vectors[text] = encode_text([text])[0]
    return _text_vectors[text]


def search_text(query: str, crop: str | None = None, k: int = 8) -> list[dict]:
    """Keyword overlap over approved records. This is what the answer uses."""
    q = _tokens(query)
    scored = []
    for rec in store.approved_records():
        if crop and rec.get("crop") and rec["crop"].lower() != crop.lower():
            continue
        doc = " ".join([
            rec.get("condition", ""),
            rec.get("symptoms", ""),
            rec.get("distinguishing_features", ""),
            " ".join(rec.get("synonyms", [])),
        ])
        overlap = q & _tokens(doc)
        if overlap:
            scored.append((len(overlap), rec, sorted(overlap)))
    scored.sort(key=lambda x: -x[0])
    return [
        {
            "id": rec["record_id"],
            "type": "text",
            "title": rec["condition"],
            "condition": rec["condition"],
            "record_id": rec["record_id"],
            "source": rec.get("source_id"),
            "source_id": rec.get("source_id"),
            "status": rec["review_status"],
            "review_status": rec["review_status"],
            "channel": "text_keyword",
            "matched_terms": terms,
        }
        for _, rec, terms in scored[:k]
    ]


def search_semantic(query: str, crop: str | None = None, k: int = 8) -> list[dict]:
    """Cosine similarity between the question and each record's text, CLIP text encoder."""
    recs = [r for r in store.approved_records()
            if not crop or not r.get("crop") or r["crop"].lower() == crop.lower()]
    if not recs or not query.strip():
        return []
    q = encode_text([query])[0]
    scored = [(float(q @ _text_vector(_record_text(r))), r) for r in recs]
    scored.sort(key=lambda x: -x[0])
    return [{
        "id": r["record_id"],
        "type": "text",
        "title": r["condition"],
        "condition": r["condition"],
        "record_id": r["record_id"],
        "source": r.get("source_id"),
        "source_id": r.get("source_id"),
        "status": r["review_status"],
        "review_status": r["review_status"],
        "channel": "text_semantic",
        "score": round(s, 3),
    } for s, r in scored[:k]]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: an item scores 1/(k+rank) in each list it appears in."""
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            score[rid] += 1.0 / (k + rank)
    return [rid for rid, _ in sorted(score.items(), key=lambda x: -x[1])]


def search_image_to_text(image_paths: list[str], crop: str | None = None, k: int = 8) -> list[dict]:
    """Match upload image embeddings against approved record text in CLIP space."""
    recs = [r for r in store.approved_records()
            if not crop or not r.get("crop") or r["crop"].lower() == crop.lower()]
    if not recs or not image_paths:
        return []
    vectors = [embed_image_file(p) for p in image_paths]
    q = np.mean(np.stack(vectors), axis=0)
    q = q / (np.linalg.norm(q) + 1e-9)
    scored = [(float(q @ _text_vector(_record_text(r))), r) for r in recs]
    scored.sort(key=lambda x: -x[0])
    return [{
        "id": r["record_id"],
        "type": "text",
        "title": r["condition"],
        "condition": r["condition"],
        "record_id": r["record_id"],
        "source": r.get("source_id"),
        "source_id": r.get("source_id"),
        "status": r["review_status"],
        "review_status": r["review_status"],
        "channel": "image_to_text",
        "score": round(s, 3),
    } for s, r in scored[:k]]


def search_channels(
    query: str,
    crop: str | None = None,
    k: int = 8,
    image_paths: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """Keyword + semantic (+ optional image-to-text). Failures must not break the answer."""
    keyword = search_text(query, crop=crop, k=k)
    semantic: list[dict] = []
    image_text: list[dict] = []
    errors: list[str] = []
    try:
        semantic = search_semantic(query, crop=crop, k=k)
    except Exception as e:
        errors.append(f"semantic: {type(e).__name__}: {e}")
    if image_paths:
        try:
            image_text = search_image_to_text(image_paths, crop=crop, k=k)
        except Exception as e:
            errors.append(f"image_to_text: {type(e).__name__}: {e}")
    return keyword, {
        "semantic": semantic,
        "image_to_text": image_text,
        "error": "; ".join(errors) if errors else None,
    }


def shadow_compare(query: str, crop: str | None = None) -> dict:
    """Compute keyword, semantic and hybrid rankings. Only logged; the answer does not use them."""
    kw = search_text(query, crop=crop, k=8)
    kw_ids = [e["id"] for e in kw]
    sem = search_semantic(query, crop=crop, k=8)
    sem_ids = [s["id"] for s in sem]
    hybrid_ids = rrf([kw_ids, sem_ids])[:5]
    titles = {r["record_id"]: r["condition"] for r in store.approved_records()}
    return {
        "method": "shadow: the answer uses keyword only",
        "keyword": [{"id": i, "title": titles.get(i)} for i in kw_ids],
        "semantic": [s for s in sem[:5]],
        "hybrid": [{"id": i, "title": titles.get(i)} for i in hybrid_ids],
    }