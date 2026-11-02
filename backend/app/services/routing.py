"""Deterministic request routing: image assessment, text Q&A, or case follow-up."""
import re

MANAGEMENT = re.compile(
    r"\b(prevent|prevention|managing|management|control|treat|treatment|fertiliz|advice)\b",
    re.I,
)
NEW_SYMPTOM = re.compile(
    r"\b(now|changed|worse|new photo|these (are )?photos|looks different)\b",
    re.I,
)


def prior_images(case: dict) -> list[str]:
    paths: list[str] = []
    for turn in case.get("turns") or []:
        for p in turn.get("images") or []:
            if p and p not in paths:
                paths.append(p)
    return paths


def route(case: dict, text: str, new_image_count: int) -> str:
    """Return image_assessment, text_qa, or followup."""
    if new_image_count:
        return "image_assessment"
    has_history = bool(case.get("turns"))
    if has_history:
        if text and MANAGEMENT.search(text) and not NEW_SYMPTOM.search(text):
            return "text_qa"
        return "followup"
    return "text_qa" if text else "image_assessment"
