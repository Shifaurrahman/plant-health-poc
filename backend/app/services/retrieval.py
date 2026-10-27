import re

from app.store import store

# Generic words that appear in almost every record or plant_part field.
# Without these, every record matches on "leaf" alone.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "on", "in", "is", "are", "with", "my", "this",
    "that", "it", "to", "for", "some", "have", "has", "lower", "upper", "leaves", "leaf",
    "spot", "spots", "plant", "brown", "about", "weeks", "week", "days", "day",
}
TOKEN = re.compile(r"[a-z]+")


def _tokens(s: str) -> set[str]:
    return {t for t in TOKEN.findall(s.lower()) if t not in STOPWORDS}


def search_text(query: str, crop: str | None = None, k: int = 8) -> list[dict]:
    """Keyword overlap over approved records. Image matching is in image_retrieval.py."""
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
            "source": rec.get("source_id"),
            "status": rec["review_status"],
            "matched_terms": terms,
        }
        for _, rec, terms in scored[:k]
    ]