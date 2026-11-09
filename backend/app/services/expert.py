"""Expert decision layer.

Combines photo evidence and text evidence into one verdict, using rules that are
written here and checked against expert-approved records. No language model decides
the verdict. Models only provide the evidence (photo votes, selected text records).

Verdicts:
  likely              photo and farmer description both support one cause
  possible            more than one cause is supported, or the photo alone supports one
  cannot_determine    evidence is missing, too weak, or disagrees
"""
from collections import Counter

from app.config import HEALTHY_CLASS, MAJORITY_SHARE, PHOTO_MIN_VOTES
from app.store import store

MIN_CLOSE_VOTE_GAP = 2  # a second candidate with this few votes or fewer is not a real rival


def _checks(condition: str) -> list[str]:
    rec = next((r for r in store.approved_records() if r["condition"] == condition), None)
    raw = (rec or {}).get("confirm_by") or []
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(";") if x.strip()]
    return raw


def _result(verdict, candidates, reasons, checks, text_set=()):
    seen = []
    for c in checks:
        if c not in seen:
            seen.append(c)
    return {
        "verdict": verdict,
        "candidates": candidates,
        "reasons": reasons,
        "checks_needed": seen[:5],
        "text_conditions": sorted(text_set),
    }


def decide(photo_votes: Counter, n_photo: int, text_conditions: list[str], has_text: bool) -> dict:
    """photo_votes: class -> count among photo matches that cleared the similarity floor.
    n_photo: number of neighbours that voted. text_conditions: conditions the text step selected.
    has_text: the farmer typed a description."""
    reasons = []

    if n_photo == 0 and not text_conditions:
        return _result("cannot_determine", [], ["No photo or description evidence was available."], [])

    ranked = photo_votes.most_common()
    top_cls, top_n = ranked[0] if ranked else (None, 0)
    share = top_n / n_photo if n_photo else 0.0
    photo_supports_top = bool(top_cls) and n_photo >= PHOTO_MIN_VOTES and share >= MAJORITY_SHARE

    if top_cls and photo_supports_top:
        reasons.append(f"{top_n} of {n_photo} nearest reference photos are {top_cls}.")
    elif n_photo:
        reasons.append("The nearest reference photos are split between several classes.")

    text_set = set(text_conditions)
    text_agrees = bool(top_cls) and top_cls in text_set
    text_disagrees = has_text and bool(text_set) and not text_agrees

    # Healthy is never a disease candidate. A photo that looks healthy is recorded as such.
    if photo_supports_top and top_cls == HEALTHY_CLASS and not text_set:
        reasons.append("The photo looks like healthy reference photos.")
        return _result("possible", [top_cls], reasons, _checks(top_cls), text_set)

    # Rule 1: photo and description point to different causes -> do not choose.
    if photo_supports_top and text_disagrees:
        reasons.append("The farmer's description points to different causes from the photo.")
        checks = _checks(top_cls) + [c for t in text_set for c in _checks(t)]
        return _result("cannot_determine", [], reasons, checks, text_set)

    # Rule 2: photo and description agree -> likely, unless a described rival is also close.
    if photo_supports_top and text_agrees:
        rivals = [c for c, n in ranked[1:] if n >= MIN_CLOSE_VOTE_GAP and c in text_set]
        cands = [top_cls] + rivals
        reasons.append("The farmer's description matches the same cause.")
        if rivals:
            reasons.append(f"{', '.join(rivals)} is also described and cannot be separated from the photo.")
            checks = [c for x in cands for c in _checks(x)]
            return _result("possible", cands, reasons, checks, text_set)
        return _result("likely", [top_cls], reasons, _checks(top_cls), text_set)

    # Rule 3: photo alone supports a cause -> at most possible.
    if photo_supports_top and not has_text:
        reasons.append("Only the photo was checked. A description or close-up is needed to confirm.")
        return _result("possible", [top_cls], reasons, _checks(top_cls), text_set)

    # Rule 4: text alone, no clear photo result -> possible for one match, otherwise cannot_determine.
    if text_set and not photo_supports_top:
        reasons.append("The description matches these records, but the photo does not clearly agree.")
        checks = [c for t in text_set for c in _checks(t)]
        verdict = "possible" if len(text_set) == 1 else "cannot_determine"
        return _result(verdict, sorted(text_set), reasons, checks, text_set)

    # Rule 5: nothing decisive.
    reasons.append("The evidence does not support a single cause.")
    checks = [c for name, _ in ranked[:3] for c in _checks(name)]
    return _result("cannot_determine", [c for c, _ in ranked[:3]], reasons, checks, text_set)