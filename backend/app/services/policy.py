"""Answer-safety rules.

Chemical product names, doses and spray intervals are removed from answers unless
the same text appears in an approved record (written by the expert).
The chemical list is a starting point; the expert should extend it.
"""
import re

from app.store import store

CHEMICAL_TERMS = {
    "mancozeb", "copper oxychloride", "carbendazim", "chlorpyrifos", "imidacloprid",
    "cypermethrin", "glyphosate", "propiconazole", "hexaconazole", "thiamethoxam",
    "dimethoate", "malathion", "fipronil", "azoxystrobin", "difenoconazole", "metalaxyl",
}
DOSE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:(?:ml|mls|l|litres?|liters?|g|gms?|grams?|kg|mg|ppm)\b|%|percent\b)"
    r"(?:\s*(?:per|/)\s*(?:l|litres?|liters?|ha|acre|kg|plant|tank|day)\b)?",
    re.IGNORECASE,
)
INTERVAL = re.compile(
    r"\bevery\s+\d+\s*(?:days?|weeks?)\b|\b\d+\s*(?:days?|weeks?)\s+(?:interval|apart|gap)\b",
    re.IGNORECASE,
)
REMOVED_MARKER = "[removed: not in reviewed records]"


def _approved_text() -> str:
    parts = []
    for r in store.approved_records():
        for key in ("management_general", "symptoms", "distinguishing_features", "next_step", "progression", "context"):
            parts.append(str(r.get(key) or ""))
        parts.extend(r.get("limitations") or [])
    return " ".join(parts).lower()


def scrub(text: str) -> tuple[str, int]:
    """Remove unsupported chemical names, doses and intervals. Returns (clean_text, removed_count)."""
    if not text:
        return text, 0
    allowed = _approved_text()
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        if match.group(0).lower() in allowed:
            return match.group(0)
        count += 1
        return REMOVED_MARKER

    for term in CHEMICAL_TERMS:
        text = re.sub(re.escape(term), repl, text, flags=re.IGNORECASE)
    text = DOSE.sub(repl, text)
    text = INTERVAL.sub(repl, text)
    return text, count