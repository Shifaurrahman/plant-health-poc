"""Text evidence for the farmer's typed words, and image-to-text matching.

Text evidence path (search_channels):
  1. Embedding shortlist: the typed words are compared with each approved record's text.
  2. Relevance: a language model picks which shortlisted records match the described symptoms
     (services/relevance.py). If it is off or fails, the embedding score threshold is used.

No word lists or keyword overlap decide evidence. The keyword search is kept for reference only,
and search_channels does not return its results.

Channels returned:
  text_semantic  a shortlisted record selected by relevance (or by score when relevance is off)
  image_to_text  photo compared with record text; off unless PH_IMAGE_TO_TEXT=yes
"""
import os
from collections import defaultdict

import numpy as np

from app.config import RELEVANCE_SHORTLIST_K, SEMANTIC_MIN_SCORE
from app.services import relevance
from app.services.embedder import embed_image_file, encode_document, encode_query
from app.store import store

# Image-to-text stays off until its scores are calibrated on held-out photos.
IMAGE_TO_TEXT_ENABLED = os.getenv("PH_IMAGE_TO_TEXT", "no").lower() == "yes"

_text_vectors: dict[str, object] = {}  # record text -> embedding (computed once per text)


def _record_text(rec: dict) -> str:
    return " ".join(x for x in [rec.get("condition", ""), rec.get("symptoms", ""),
                                rec.get("distinguishing_features", "")] if x)


def _text_vector(text: str):
    if text not in _text_vectors:
        _text_vectors[text] = encode_document([text])[0]
    return _text_vectors[text]


def _record_hit(rec: dict, channel: str, score: float | None = None) -> dict:
    hit = {
        "id": rec["record_id"],
        "type": "text",
        "title": rec["condition"],
        "condition": rec["condition"],
        "record_id": rec["record_id"],
        "source": rec.get("source_id"),
        "source_id": rec.get("source_id"),
        "status": rec["review_status"],
        "review_status": rec["review_status"],
        "channel": channel,
    }
    if score is not None:
        hit["score"] = round(score, 3)
    return hit


def _crop_match(rec: dict, crop: str | None) -> bool:
    return not crop or not rec.get("crop") or rec["crop"].lower() == crop.lower()


def search_text(query: str, crop: str | None = None, k: int = 8) -> list[dict]:
    """Keyword overlap. Kept for reference and debugging. Not used as evidence."""
    import re
    tok = lambda s: set(re.findall(r"[a-z]+", s.lower()))  # noqa: E731
    q = tok(query)
    if not q:
        return []
    scored = []
    for rec in store.approved_records():
        if not _crop_match(rec, crop):
            continue
        overlap = q & tok(_record_text(rec) + " " + " ".join(rec.get("synonyms", [])))
        if overlap:
            scored.append((len(overlap), rec, sorted(overlap)))
    scored.sort(key=lambda x: -x[0])
    out = []
    for _, rec, terms in scored[:k]:
        hit = _record_hit(rec, "text_keyword")
        hit["matched_terms"] = terms
        out.append(hit)
    return out


def search_semantic(query: str, crop: str | None = None, k: int = 8) -> list[dict]:
    """Embedding similarity between the typed words and each record's text."""
    recs = [r for r in store.approved_records() if _crop_match(r, crop)]
    if not recs or not query.strip():
        return []
    q = encode_query([query])[0]
    scored = [(float(q @ _text_vector(_record_text(r))), r) for r in recs]
    scored.sort(key=lambda x: -x[0])
    return [_record_hit(r, "text_semantic", s) for s, r in scored[:k]]


def select_text_evidence(user_text: str, crop: str | None, k: int) -> tuple[list[dict], str | None]:
    """Shortlist by embedding, then let the relevance step pick which ones match."""
    if not user_text.strip():
        return [], None
    shortlist = [h for h in search_semantic(user_text, crop=crop, k=k)
                 if h["score"] >= SEMANTIC_MIN_SCORE]
    if not shortlist:
        return [], None

    by_id = {r["record_id"]: r for r in store.approved_records()}
    items = [{
        "id": h["id"],
        "condition": h["condition"],
        "symptoms": (by_id.get(h["id"]) or {}).get("symptoms", ""),
        "distinguishing_features": (by_id.get(h["id"]) or {}).get("distinguishing_features", ""),
    } for h in shortlist]

    ids, err = relevance.select_relevant(user_text, items)
    if ids is None:  # relevance off or failed: fall back to the embedding shortlist
        return shortlist, err
    chosen = [dict(h, matched_by="model") for h in shortlist if h["id"] in set(ids)]
    return chosen, None


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: an item scores 1/(k+rank) in each list it appears in."""
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            score[rid] += 1.0 / (k + rank)
    return [rid for rid, _ in sorted(score.items(), key=lambda x: -x[1])]


def search_image_to_text(image_paths: list[str], crop: str | None = None, k: int = 8) -> list[dict]:
    """Photo compared with record text in the shared embedding space. Off by default."""
    recs = [r for r in store.approved_records() if _crop_match(r, crop)]
    if not recs or not image_paths:
        return []
    vectors = [embed_image_file(p) for p in image_paths]
    q = np.mean(np.stack(vectors), axis=0)
    q = q / (np.linalg.norm(q) + 1e-9)
    scored = [(float(q @ _text_vector(_record_text(r))), r) for r in recs]
    scored.sort(key=lambda x: -x[0])
    return [_record_hit(r, "image_to_text", s) for s, r in scored[:k]]


def search_channels(
    query: str,
    crop: str | None = None,
    k: int = 8,
    image_paths: list[str] | None = None,
    user_text: str = "",
) -> tuple[list[dict], dict]:
    """Same return shape as before, so pipeline.py does not change.

    Returns (keyword_list, diag). keyword_list is always empty now: text evidence is in
    diag["semantic"], selected by the model or by score. The pipeline already reads that list.
    """
    errors: list[str] = []

    semantic: list[dict] = []
    try:
        semantic, err = select_text_evidence(user_text, crop, RELEVANCE_SHORTLIST_K)
        if err:
            errors.append(err)
    except Exception as e:  # fail soft: the photo answer must still be returned
        errors.append(f"semantic: {type(e).__name__}: {e}")

    image_text: list[dict] = []
    if image_paths and IMAGE_TO_TEXT_ENABLED:
        try:
            image_text = [h for h in search_image_to_text(image_paths, crop=crop, k=k)
                          if h["score"] >= SEMANTIC_MIN_SCORE]
        except Exception as e:
            errors.append(f"image_to_text: {type(e).__name__}: {e}")

    return [], {
        "semantic": semantic,
        "image_to_text": image_text,
        "error": "; ".join(errors) if errors else None,
    }