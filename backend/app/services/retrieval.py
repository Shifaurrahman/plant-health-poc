"""Text and image-to-text retrieval over approved records.

Channels:
  keyword        word overlap with the farmer's typed words (plus case context)
  text_semantic  embedding similarity between the typed question and record text
  image_to_text  embedding similarity between the photo and record text

Gating rules (enforced by search_channels):
  - semantic runs only when the farmer typed a question
  - semantic and image_to_text hits below SEMANTIC_MIN_SCORE are dropped
"""
import re
from collections import defaultdict

import numpy as np

from app.config import SEMANTIC_MIN_SCORE
from app.services.embedder import embed_image_file, encode_document, encode_query
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


def _above_min(hits: list[dict]) -> list[dict]:
    return [h for h in hits if h.get("score", 1.0) >= SEMANTIC_MIN_SCORE]


def search_text(query: str, crop: str | None = None, k: int = 8) -> list[dict]:
    """Keyword overlap over approved records. This is what the answer uses."""
    q = _tokens(query)
    if not q:
        return []
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
    out = []
    for _, rec, terms in scored[:k]:
        hit = _record_hit(rec, "text_keyword")
        hit["matched_terms"] = terms
        out.append(hit)
    return out


def search_semantic(query: str, crop: str | None = None, k: int = 8) -> list[dict]:
    """Cosine similarity between the typed question and each record's text (query side of the embedder)."""
    recs = [r for r in store.approved_records()
            if not crop or not r.get("crop") or r["crop"].lower() == crop.lower()]
    if not recs or not query.strip():
        return []
    q = encode_query([query])[0]
    scored = [(float(q @ _text_vector(_record_text(r))), r) for r in recs]
    scored.sort(key=lambda x: -x[0])
    return [_record_hit(r, "text_semantic", s) for s, r in scored[:k]]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: an item scores 1/(k+rank) in each list it appears in."""
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            score[rid] += 1.0 / (k + rank)
    return [rid for rid, _ in sorted(score.items(), key=lambda x: -x[1])]


def search_image_to_text(image_paths: list[str], crop: str | None = None, k: int = 8) -> list[dict]:
    """Match upload image embeddings against approved record text in the shared embedding space."""
    recs = [r for r in store.approved_records()
            if not crop or not r.get("crop") or r["crop"].lower() == crop.lower()]
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
    """Keyword + semantic + image-to-text, with gating.

    query      text used for keyword and semantic matching (may include case context)
    user_text  what the farmer actually typed; semantic search runs only if this is non-empty
    Failures must not break the answer: each channel fails soft.
    """
    keyword = search_text(query, crop=crop, k=k) if query.strip() else []
    semantic: list[dict] = []
    image_text: list[dict] = []
    errors: list[str] = []

    if user_text.strip() and query.strip():
        try:
            semantic = _above_min(search_semantic(query, crop=crop, k=k))
        except Exception as e:
            errors.append(f"semantic: {type(e).__name__}: {e}")

    if image_paths:
        try:
            image_text = _above_min(search_image_to_text(image_paths, crop=crop, k=k))
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
        "semantic": sem[:5],
        "hybrid": [{"id": i, "title": titles.get(i)} for i in hybrid_ids],
    }