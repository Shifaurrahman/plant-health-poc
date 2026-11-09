"""Model-based selection of text records for the farmer's words.

The embedding model gives a shortlist. This module asks a language model which shortlisted
records describe the symptoms the farmer actually described. The model may only return IDs
from the shortlist, so it cannot introduce new evidence.

Returns (ids, error):
  ids is None when the model was not used, so the caller falls back to embedding scores.
"""
import json
import logging

from app.config import RELEVANCE_MODEL, RELEVANCE_PROVIDER, RELEVANCE_TIMEOUT_S

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You select knowledge-base records for a farmer's description of a brinjal (eggplant) plant.

Rules:
- Choose a record only if the symptoms in the record match what the farmer described.
- Do not choose a record because of a shared common word (colour, leaf, time, place, or similar).
- If no record matches the description, return an empty list.
- Use only the record IDs given to you. Do not invent IDs, diagnoses or facts.

Reply with JSON only, in this exact form:
{"relevant_ids": ["<id>", ...]}"""


def select_relevant(farmer_text: str, shortlist: list[dict]) -> tuple[list[str] | None, str | None]:
    """shortlist items need: id, condition, symptoms, distinguishing_features."""
    if RELEVANCE_PROVIDER == "none" or not shortlist or not farmer_text.strip():
        return None, None
    if RELEVANCE_PROVIDER != "anthropic":
        return None, f"unknown PH_RELEVANCE_PROVIDER '{RELEVANCE_PROVIDER}'"

    allowed = {r["id"] for r in shortlist}
    payload = {
        "farmer_text": farmer_text,
        "records": [
            {
                "id": r["id"],
                "condition": r.get("condition"),
                "symptoms": r.get("symptoms") or "",
                "distinguishing_features": r.get("distinguishing_features") or "",
            }
            for r in shortlist
        ],
    }
    try:
        import anthropic

        client = anthropic.Anthropic(timeout=RELEVANCE_TIMEOUT_S)
        msg = client.messages.create(
            model=RELEVANCE_MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        ids = data.get("relevant_ids", [])
        if not isinstance(ids, list):
            raise ValueError("relevant_ids is not a list")
        # Only IDs from the shortlist are accepted.
        return [i for i in ids if i in allowed], None
    except Exception as e:  # the caller falls back to embedding scores
        logger.warning("relevance model failed: %s", e)
        return None, f"relevance model: {type(e).__name__}: {e}"