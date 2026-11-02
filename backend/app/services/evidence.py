"""Merge retrieval channels and pick a small diverse evidence set for synthesis."""
from app.config import EVIDENCE_K


def _condition(item: dict) -> str:
    return (item.get("condition") or item.get("title") or "").lower()


def select_evidence(channels: list[list[dict]], k: int = EVIDENCE_K) -> list[dict]:
    """Dedupe by id, then take diverse conditions and types. Repeated copies of one source count once."""
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    merged: list[dict] = []
    for channel in channels:
        for item in channel:
            eid = item.get("id")
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            src = item.get("source_id") or item.get("source")
            extra = dict(item)
            extra["independent_source"] = bool(src) and src not in seen_sources
            if src:
                seen_sources.add(src)
            merged.append(extra)

    picked: list[dict] = []
    used_conditions: set[str] = set()
    used_types: dict[str, int] = {"image": 0, "text": 0}

    def take(pred) -> None:
        for item in merged:
            if len(picked) >= k:
                return
            if item in picked:
                continue
            if pred(item):
                picked.append(item)
                used_conditions.add(_condition(item))
                used_types[item.get("type") or "text"] = used_types.get(item.get("type") or "text", 0) + 1

    take(lambda i: i.get("type") == "image" and _condition(i) not in used_conditions)
    take(lambda i: i.get("type") == "text" and _condition(i) not in used_conditions)
    take(lambda i: _condition(i) not in used_conditions)
    take(lambda i: used_types.get(i.get("type") or "text", 0) < 4)
    take(lambda _i: True)
    return picked[:k]
