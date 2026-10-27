def validate(draft: dict, evidence: list[dict]) -> dict:
    """Drop cited evidence IDs that were not actually retrieved. Flag causes left without support."""
    valid_ids = {e["id"] for e in evidence}
    for cause in draft["candidate_causes"]:
        cause.setdefault("against_or_missing", [])
        cause["evidence_ids"] = [i for i in cause.get("evidence_ids", []) if i in valid_ids]
        if not cause["evidence_ids"]:
            cause["against_or_missing"].append("No valid reviewed evidence cited for this cause")
    draft["questions"] = draft.get("questions", [])[:3]
    return draft
