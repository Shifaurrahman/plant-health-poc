from app.services import image_retrieval, retrieval, validator
from app.services.vlm import get_vlm

LIMITATIONS = [
    "Decision support only. Visible symptoms can have several causes, including pests, nutrition or environment.",
    "Image matching finds visually similar reference photos. It does not confirm a diagnosis.",
    "Some causes need physical inspection or laboratory testing to confirm.",
    "Coverage is never marked strong automatically in the POC; an expert must judge it.",
]
STUB_NOTE = "Visual analysis is not configured, so the observations are placeholders."


def fallback_draft(image_candidates: list[dict], text_evidence: list[dict], has_images: bool, meta: dict) -> dict:
    causes = list(image_candidates[:3])
    names = {c["name"] for c in causes}
    for e in text_evidence:
        if len(causes) >= 3:
            break
        if e["title"] not in names:
            causes.append({
                "name": e["title"],
                "support": [f"Matched terms: {', '.join(e['matched_terms'])}"],
                "against_or_missing": ["Text match only; not checked against the photo"],
                "evidence_ids": [e["id"]],
            })
            names.add(e["title"])

    questions = []
    if not has_images:
        questions.append("Can you upload a whole-plant photo and a close-up of an affected leaf?")
    if not meta.get("crop"):
        questions.append("Which crop is this?")

    next_step = (
        "Compare the photo with the cited reference photos and confirm with a local plant health expert."
        if causes else
        "No approved reference photos match. Consult a local plant health adviser."
    )
    return {
        "candidate_causes": causes,
        "alternatives": [c["name"] for c in image_candidates[3:6]],
        "questions": questions,
        "next_step": next_step,
    }


def run_assessment(case: dict, text: str, image_paths: list[str]) -> dict:
    meta = case["meta"]
    vlm = get_vlm()
    observations = vlm.observe(image_paths, text, meta)

    limitations = list(LIMITATIONS)
    if vlm.name == "stub":
        limitations.append(STUB_NOTE)

    matches: list[dict] = []
    for p in image_paths:
        try:
            matches.extend(image_retrieval.match_image(p))
        except Exception as e:  # missing index or unreadable image: report it, don't crash the request
            limitations.append(f"Image matching failed for one photo: {e}")

    query = " ".join(filter(None, [text, meta.get("crop"), meta.get("plant_part"), meta.get("symptom_duration")]))
    text_evidence = retrieval.search_text(query, crop=meta.get("crop"))
    image_candidates = image_retrieval.candidates_from_matches(matches)
    evidence = image_retrieval.evidence_from_matches(matches) + text_evidence

    draft = vlm.synthesize(meta, text, observations, evidence)
    if draft is None:
        draft = fallback_draft(image_candidates, text_evidence, bool(image_paths), meta)
    draft = validator.validate(draft, evidence)

    return {
        "observations": observations,
        "candidate_causes": draft["candidate_causes"],
        "alternatives": draft["alternatives"],
        "coverage": "none" if not evidence else "partial",
        "limitations": limitations,
        "questions": draft["questions"],
        "next_step": draft["next_step"],
        "retrieved_evidence": evidence,
        "model_version": f"{vlm.name}+clip-retrieval+keyword-0.2",
    }