import time
from collections import Counter

from app.config import (
    CHANNEL_K, CROP, HEALTHY_CLASS, MAJORITY_SHARE, MAX_ALTERNATIVES,
    MAX_CANDIDATE_CAUSES, MAX_QUESTIONS, MAX_RETRIEVAL_ROUNDS, PHOTO_MIN_SIMILARITY,
    PHOTO_MIN_VOTES, SUPPORTED_CROPS, crops_label,
)
from app.services import expert, image_retrieval, retrieval, validator
from app.services.evidence import select_evidence
from app.services.photo_guidance import quality_questions
from app.services.routing import prior_images, route
from app.services.versions import current_versions
from app.services.vlm import get_vlm
from app.store import store

BASE_LIMITATIONS = [
    "Decision support only. Visible symptoms can have infectious, pest-related or environmental causes.",
    "Image matching finds visually similar reference photos. It does not confirm a diagnosis.",
    "Some causes need physical inspection or laboratory testing to confirm.",
    "Sources support facts in the knowledge base; they do not prove the uploaded plant has that condition.",
    "Coverage is never marked strong automatically in the POC; an expert must judge it.",
]
STUB_NOTE = "Visual analysis is not configured, so the observations are placeholders."
WEAK_PHOTO_NOTE = "No reference photo was similar enough to the upload to compare."

OUTCOMES = {
    "match": {
        "next_step": "Compare the photo with the cited reference photos and confirm with a local plant health expert.",
        "limitation": None,
    },
    "healthy_appearance": {
        "next_step": "The photo looks like healthy reference photos. If you still see symptoms, send a close-up of both leaf surfaces and a whole-plant photo.",
        "limitation": "A healthy-looking photo does not rule out early symptoms, pests or nutrient problems.",
    },
    "unknown_cause": {
        "next_step": "The photo does not clearly match one reference class. Send a close-up of both leaf surfaces and a whole-plant photo.",
        "limitation": "The nearest reference photos are split between several classes, so no single cause is suggested.",
    },
    "insufficient_image": {
        "next_step": "Upload a sharper, larger photo of the affected leaf, in focus and filling the frame.",
        "limitation": "The photo is too small or too plain to compare with reference photos.",
    },
    "unsupported_crop": {
        "next_step": None,  # filled from config at runtime
        "limitation": "No photo matching is done for crops other than the supported set.",
    },
    "text_only_match": {
        "next_step": "Text matches only. Upload a photo of the affected leaf to compare with reference photos.",
        "limitation": "Text matching cannot see the plant. Model knowledge is labelled separately from reviewed records.",
    },
    "conflict": {
        "next_step": "The photo and the reference records disagree, so no cause is suggested. Send a close-up of both leaf surfaces and a whole-plant photo, and ask a local plant health adviser.",
        "limitation": "The visible symptoms do not fit the nearest reference class, so that class is not suggested.",
    },
    "no_match": {
        "next_step": "No approved reference material matches. Consult a local plant health adviser.",
        "limitation": "The knowledge base has no matching reviewed material for this question.",
    },
}


def decide_outcome(crop_ok: bool, image_paths: list[str], quality_issues: list[list[str]],
                   matches: list[dict], text_evidence: list[dict], raw_match_count: int = 0) -> str:
    """Photo-quality gate. Only the unsupported_crop and insufficient_image results are used from it."""
    if not crop_ok:
        return "unsupported_crop"
    if image_paths and all(quality_issues):
        return "insufficient_image"
    if not matches:
        if raw_match_count:
            return "unknown_cause"
        return "text_only_match" if text_evidence else "no_match"
    if len(matches) < PHOTO_MIN_VOTES:
        return "unknown_cause"
    counts = Counter(m["class"] for m in matches)
    label, n = counts.most_common(1)[0]
    if n / len(matches) < MAJORITY_SHARE:
        return "unknown_cause"
    if label == HEALTHY_CLASS:
        return "healthy_appearance"
    return "match"


def verdict_to_outcome(verdict: dict, healthy_photo: bool) -> str:
    """Map the expert verdict to the outcome label the UI and the response use."""
    if healthy_photo:
        return "healthy_appearance"
    v = verdict["verdict"]
    if v == "likely":
        return "match"
    if v == "possible":
        return "unknown_cause"
    if any("different causes" in r for r in verdict["reasons"]):
        return "conflict"
    return "unknown_cause"


def apply_fit_check(draft: dict, outcome: str) -> tuple[str, dict]:
    """Use the model's per-candidate fit. All candidates poor -> conflict. Otherwise drop poor-fit ones."""
    causes = draft.get("candidate_causes") or []
    if causes and all(c.get("fit") == "poor" for c in causes):
        draft["rejected_candidates"] = [c["name"] for c in causes]
        draft["candidate_causes"] = []
        return "conflict", draft
    if any(c.get("fit") == "poor" for c in causes):
        draft["rejected_candidates"] = [c["name"] for c in causes if c.get("fit") == "poor"]
        draft["candidate_causes"] = [c for c in causes if c.get("fit") != "poor"]
    return outcome, draft


def _outcome_next_step(outcome: str) -> str:
    if outcome == "unsupported_crop":
        return (
            f"This version supports {crops_label()} only. "
            "Consult a local plant health adviser for other crops."
        )
    return OUTCOMES[outcome]["next_step"] or "Confirm with a local plant health expert."


def fallback_draft(image_candidates, text_evidence, has_images, meta, outcome, extra_questions) -> dict:
    causes = list(image_candidates[:MAX_CANDIDATE_CAUSES])
    names = {c["name"] for c in causes}
    for e in text_evidence:
        if len(causes) >= MAX_CANDIDATE_CAUSES:
            break
        if e["title"] not in names:
            causes.append({
                "name": e["title"],
                "support": [f"Matched words: {', '.join(e.get('matched_terms') or [])}"]
                if e.get("matched_terms") else ["Retrieved from reviewed records"],
                "against_or_missing": ["Text match only; not checked against the photo"],
                "evidence_ids": [e["id"]],
            })
            names.add(e["title"])

    questions = list(extra_questions)
    if not has_images and not questions:
        questions.append("Can you upload a whole-plant photo and a close-up of an affected leaf? "
                         "Those views show distribution and lesion pattern.")
    if not meta.get("crop"):
        questions.append("Which crop is this? Crop identity is treated as uncertain until you confirm it.")

    alt_start = MAX_CANDIDATE_CAUSES
    alt_end = MAX_CANDIDATE_CAUSES + MAX_ALTERNATIVES
    return {
        "candidate_causes": causes,
        "alternatives": [c["name"] for c in image_candidates[alt_start:alt_end]],
        "questions": questions[:MAX_QUESTIONS],
        "next_step": _outcome_next_step(outcome),
    }


def record_limitations(matches: list[dict]) -> list[str]:
    seen, out = set(), []
    for m in matches:
        for lim in m["record"].get("limitations") or []:
            if lim not in seen:
                seen.add(lim)
                out.append(lim)
    return out


def _reference_text(record_id: str | None) -> str:
    """The expert's written description of a condition. This is all the model may describe."""
    rec = store.get_record(record_id) if record_id else None
    if not rec:
        return ""
    parts = [rec.get("symptoms") or "", rec.get("distinguishing_features") or ""]
    return " ".join(x.strip() for x in parts if x and x.strip())


def _enrich(evidence: list[dict]) -> list[dict]:
    """Attach source details, a normalised review status, and the reference text for grounding."""
    out = []
    for e in evidence:
        source_id = e.get("source_id") or e.get("source")
        src = store.get_source(source_id) if source_id else None
        rec = store.get_record(e.get("record_id") or e.get("id")) if (e.get("record_id") or e.get("id")) else None
        out.append({
            **e,
            "source_id": source_id,
            "review_status": e.get("review_status") or e.get("status"),
            "source_title": src["title"] if src else None,
            "publisher": src["publisher"] if src else None,
            "source_url": src["url"] if src else None,
            "published_on": src["published_on"] if src else None,
            "label_basis": e.get("label_basis") or (rec.get("label_basis") if rec else None),
            "reference_text": _reference_text(e.get("record_id") or (e.get("id") if e.get("type") == "text" else None)),
        })
    return out


def _keep_evidence(pool: list[dict], candidate_names: set[str], has_user_text: bool) -> list[dict]:
    """Decide which evidence may be shown.

    - Photo evidence is always kept (it already passed PHOTO_MIN_SIMILARITY).
    - A text record is kept if it is a candidate cause.
    - Otherwise a text record is kept only when the farmer typed words that matched it.
    """
    kept = []
    for e in pool:
        if e.get("type") == "image":
            kept.append(e)
            continue
        title = e.get("title") or e.get("condition")
        if title and title in candidate_names:
            kept.append(e)
        elif has_user_text and e.get("channel") in ("text_keyword", "text_semantic"):
            kept.append(e)
    return kept


def _crop_ok(crop: str) -> bool:
    return crop in SUPPORTED_CROPS


def _query_text(text: str, meta: dict, summary: dict) -> str:
    parts = [
        text,
        meta.get("crop"),
        meta.get("plant_part"),
        meta.get("symptom_duration"),
        meta.get("recent_watering_or_treatment"),
        " ".join(summary.get("candidates") or []) if summary else "",
    ]
    return " ".join(filter(None, parts))


def _second_round_items(query: str, crop: str | None, draft: dict, known: set[str], user_text: str) -> list[dict]:
    """At most one extra text retrieval, if synthesis named conditions we have not retrieved yet."""
    if MAX_RETRIEVAL_ROUNDS < 1 or not query.strip():
        return []
    names = " ".join(
        [c.get("name") or "" for c in draft.get("candidate_causes") or []]
        + list(draft.get("alternatives") or [])
        + list(draft.get("questions") or [])
    )
    extra_terms = [w for w in names.split() if len(w) > 4]
    if not extra_terms:
        return []
    _, more = retrieval.search_channels(
        query + " " + " ".join(extra_terms[:8]), crop=crop, k=CHANNEL_K, user_text=user_text,
    )
    return [e for e in more.get("semantic") or []
            if (e.get("condition") or e.get("title") or "").lower() not in known]


def _case_summary(meta: dict, assessment: dict) -> dict:
    return {
        "crop": meta.get("crop"),
        "region": meta.get("region"),
        "last_outcome": assessment.get("outcome"),
        "candidates": [c["name"] for c in assessment.get("candidate_causes") or []],
        "open_questions": assessment.get("questions") or [],
        "coverage": assessment.get("coverage"),
        "route": assessment.get("route"),
    }


def _coverage(candidates: list[dict], image_candidate_names: set[str], evidence: list[dict]) -> str:
    """'partial' only when a candidate has backing: an approved photo match or a text record with the same name."""
    names = {c["name"] for c in candidates}
    image_backed = bool(names & image_candidate_names)
    text_backed = any(
        e.get("type") == "text" and (e.get("title") or e.get("condition")) in names
        for e in evidence
    )
    return "partial" if (image_backed or text_backed) else "none"


def run_assessment(case: dict, text: str, image_paths: list[str], quality_issues: list[list[str]]) -> dict:
    t0 = time.perf_counter()
    meta = case["meta"]
    vlm = get_vlm()
    crop_raw = (meta.get("crop") or "").strip().lower()
    crop = crop_raw or CROP
    crop_ok = _crop_ok(crop)
    crop_uncertain = not crop_raw
    user_text = (text or "").strip()
    has_user_text = bool(user_text)
    request_route = route(case, text, len(image_paths))

    analysis_images = list(image_paths)
    analysis_quality = list(quality_issues)
    if request_route == "followup" and not analysis_images:
        analysis_images = prior_images(case)
        analysis_quality = [[] for _ in analysis_images]

    skip_vision = request_route == "text_qa" and not image_paths
    usable = [] if skip_vision else [p for p, issues in zip(analysis_images, analysis_quality) if not issues]

    observations = [] if skip_vision else vlm.observe(analysis_images, text, meta)

    # ---- photo matching, then the similarity floor ----
    t_img = time.perf_counter()
    raw_matches, errors = [], []
    if crop_ok and not skip_vision:
        for p in usable:
            try:
                raw_matches.extend(image_retrieval.match_image(p))
            except Exception as e:  # report the failure instead of failing the whole request
                errors.append(f"Image matching failed for one photo: {e}")
    matches = [m for m in raw_matches if m["similarity"] >= PHOTO_MIN_SIMILARITY]
    image_ms = round((time.perf_counter() - t_img) * 1000)

    # ---- text channels (gated on what the farmer actually typed) ----
    t_txt = time.perf_counter()
    summary = case.get("summary") or {}
    query = _query_text(user_text, meta, summary) if has_user_text else ""
    # Hard crop filter only when the user confirmed crop; otherwise soft (no filter).
    crop_filter = crop_raw if crop_raw and crop_ok else (crop if crop_ok and not crop_uncertain else None)
    keyword_ev, channel_diag = retrieval.search_channels(
        query, crop=crop_filter, k=CHANNEL_K,
        image_paths=usable or None, user_text=user_text,
    )
    semantic_pool = channel_diag.get("semantic") or []
    image_text_pool = channel_diag.get("image_to_text") or []
    text_ms = round((time.perf_counter() - t_txt) * 1000)

    # ---- expert verdict: photo votes and selected text records decide the outcome ----
    photo_votes = Counter(m["record"]["condition"] for m in matches)
    text_conditions = [e["condition"] for e in semantic_pool]
    verdict = expert.decide(photo_votes, len(matches), text_conditions, has_user_text)
    healthy_photo = verdict["candidates"] == [HEALTHY_CLASS]

    diagnostics = {
        "route": request_route,
        "photo_matches_raw": len(raw_matches),
        "photo_matches_passing": len(matches),
        "photo_min_similarity": PHOTO_MIN_SIMILARITY,
        "expert_verdict": verdict["verdict"],
        "text_channels": {
            "typed_text": has_user_text,
            "keyword_ids": [e["id"] for e in keyword_ev],
            "semantic_ids": [e["id"] for e in semantic_pool],
            "semantic_scores": [e.get("score") for e in semantic_pool],
            "image_to_text_ids": [e["id"] for e in image_text_pool],
            "image_to_text_scores": [e.get("score") for e in image_text_pool],
            "channel_error": channel_diag.get("error"),
        },
    }

    # ---- outcome: quality checks first, then the expert verdict ----
    gate = decide_outcome(
        crop_ok,
        analysis_images if not skip_vision else [],
        analysis_quality if not skip_vision else [],
        matches,
        semantic_pool,
        len(raw_matches),
    )
    if gate in ("unsupported_crop", "insufficient_image"):
        outcome = gate
    elif skip_vision:
        outcome = "text_only_match" if semantic_pool else "no_match"
    else:
        outcome = verdict_to_outcome(verdict, healthy_photo)

    # ---- candidates and first-pass evidence ----
    if outcome in ("unsupported_crop", "insufficient_image"):
        image_candidates = []
        image_ev = []
    else:
        image_candidates = image_retrieval.candidates_from_matches(matches)
        image_ev = image_retrieval.evidence_from_matches(matches)
    # Only the candidates the expert verdict keeps may appear as image candidates.
    if verdict["candidates"]:
        image_candidates = [c for c in image_candidates if c["name"] in verdict["candidates"]]
    image_names = {c["name"] for c in image_candidates}

    pools = [image_ev, keyword_ev, semantic_pool, image_text_pool]
    selected = select_evidence([_keep_evidence(p, image_names, has_user_text) for p in pools])
    evidence = _enrich(selected)

    extra_q = quality_questions(len(analysis_images), analysis_quality, outcome)
    draft = None if skip_vision else vlm.synthesize(meta, text, observations, evidence, outcome, summary)

    if draft is None:
        draft = fallback_draft(image_candidates, semantic_pool, bool(analysis_images), meta, outcome, extra_q)
    else:
        outcome, draft = apply_fit_check(draft, outcome)
        if outcome == "conflict":
            draft["next_step"] = _outcome_next_step("conflict")
        for q in extra_q:
            if q not in draft["questions"] and len(draft["questions"]) < MAX_QUESTIONS:
                draft["questions"].append(q)

        # Re-link evidence against the candidates the model actually kept.
        final_names = {c["name"] for c in draft["candidate_causes"]}
        known = {(e.get("condition") or e.get("title") or "").lower() for e in evidence}
        extra = _second_round_items(query, meta.get("crop") or crop, draft, known, user_text)
        evidence = _enrich(select_evidence(
            [_keep_evidence(p, final_names, has_user_text) for p in pools + [extra]]
        ))

    # ---- limitations ----
    limitations = list(BASE_LIMITATIONS)
    if vlm.name == "stub":
        limitations.append(STUB_NOTE)
    for msg in [vlm.config_error, *vlm.errors()]:
        if msg:
            limitations.append(msg)
    for n, issues in enumerate(quality_issues, start=1):
        for issue in issues:
            limitations.append(f"Photo {n}: {issue}")
    limitations += errors
    if raw_matches and not matches:
        limitations.append(WEAK_PHOTO_NOTE)
    if crop_uncertain:
        limitations.append(
            f"Crop was not provided; {CROP} is assumed for this pilot and treated as uncertain."
        )
    if OUTCOMES.get(outcome, {}).get("limitation"):
        limitations.append(OUTCOMES[outcome]["limitation"])
    limitations += verdict["reasons"]
    if outcome == "unsupported_crop":
        draft["next_step"] = _outcome_next_step("unsupported_crop")
    for lim in record_limitations(matches):
        if lim not in limitations:
            limitations.append(lim)
    for name in draft.get("rejected_candidates", []):
        limitations.append(f"The visible symptoms did not fit '{name}' when checked against the photo.")
    draft["limitations"] = limitations

    if verdict["checks_needed"]:
        draft["questions"] = [f"Check: {c}" for c in verdict["checks_needed"]][:MAX_QUESTIONS]

    if outcome == "match" and matches and matches[0]["record"].get("next_step"):
        draft["next_step"] = matches[0]["record"]["next_step"]

    draft, removed = validator.validate(draft, evidence)

    coverage = _coverage(draft["candidate_causes"], image_names, evidence)
    best = max((m["similarity"] for m in raw_matches), default=None)
    versions = current_versions(vlm.name, len(store.approved_records()))
    versions["model"] = vlm.model_name
    versions["prompt"] = vlm.prompt_version

    if coverage == "none" or outcome in {"no_match", "unsupported_crop"}:
        store.log_gap({
            "case_id": case.get("case_id"),
            "crop": crop,
            "outcome": outcome,
            "question": text,
            "reason": "no matching reviewed material" if outcome == "no_match" else outcome,
        })

    return {
        "route": request_route,
        "outcome": outcome,
        "expert": verdict,
        "observations": observations,
        "candidate_causes": draft["candidate_causes"],
        "alternatives": draft["alternatives"],
        "coverage": coverage,
        "signals": {
            "retrieval_relevance": {
                "best_reference_similarity": best,
                "meaning": "How close the closest approved reference photo is to the upload. Visual only, not a diagnosis.",
            },
            "evidence_coverage": coverage,
            "diagnostic_confidence": {
                "value": None,
                "meaning": "Not assessed. Needs calibration against expert-labelled cases.",
            },
        },
        "limitations": draft["limitations"],
        "questions": draft["questions"],
        "next_step": draft["next_step"],
        "retrieved_evidence": [{k: v for k, v in e.items() if k != "reference_text"} for e in evidence],
        "policy_removals": removed,
        "diagnostics": diagnostics,
        "latency_ms": {"total": round((time.perf_counter() - t0) * 1000),
                       "image_match": image_ms, "text_match": text_ms},
        "versions": versions,
        "case_summary": _case_summary(meta, {
            "outcome": outcome, "candidate_causes": draft["candidate_causes"],
            "questions": draft["questions"], "coverage": coverage, "route": request_route,
        }),
    }