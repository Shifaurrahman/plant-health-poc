from app.config import MAX_QUESTIONS
from app.services import policy


def _scrub_list(items: list[str]) -> tuple[list[str], int]:
    out, total = [], 0
    for s in items:
        clean, n = policy.scrub(s)
        out.append(clean)
        total += n
    return out, total


def validate(draft: dict, evidence: list[dict]) -> tuple[dict, int]:
    """Drop cited IDs that were not retrieved, cap questions, and scrub forbidden content.

    Returns (draft, removed_count). removed_count should be 0 in normal runs.
    """
    valid_ids = {e["id"] for e in evidence}
    removed = 0

    for cause in draft["candidate_causes"]:
        cause.setdefault("against_or_missing", [])
        cause["evidence_ids"] = [i for i in cause.get("evidence_ids", []) if i in valid_ids]
        if not cause["evidence_ids"]:
            cause["against_or_missing"].append("No valid reviewed evidence cited for this cause")
        cause["support"], n1 = _scrub_list(cause.get("support", []))
        cause["against_or_missing"], n2 = _scrub_list(cause["against_or_missing"])
        removed += n1 + n2

    draft["alternatives"], n = _scrub_list(draft.get("alternatives", []))
    removed += n
    draft["questions"], n = _scrub_list(draft.get("questions", [])[:MAX_QUESTIONS])
    removed += n
    draft["limitations"], n = _scrub_list(draft.get("limitations", []))
    removed += n
    draft["next_step"], n = policy.scrub(draft.get("next_step", ""))
    removed += n
    return draft, removed