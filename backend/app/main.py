import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import (
    APP_VERSION, CORS_ORIGINS, IMAGE_ALLOWED_EXT, MAX_IMAGES_PER_MESSAGE,
    QUEUE_WEIGHT_DISPUTE, QUEUE_WEIGHT_IMAGE_QUALITY, QUEUE_WEIGHT_MISSING_KNOWLEDGE,
    QUEUE_WEIGHT_REQUESTED, QUEUE_WEIGHT_UNREVIEWED, SUPPORTED_CROPS, UPLOAD_DIR,
    VLM_PROVIDER, public_settings,
)
from app.routes_images import router as images_router
from app.schemas import CaseCreate, CasePatch, FeedbackIn, RecordPatch, ReviewIn
from app.services.eval_suite import run_eval
from app.services.photo_guidance import capture_guidance
from app.services.pipeline import run_assessment
from app.services.upload_validation import UploadError, validate_image
from app.store import store

app = FastAPI(title="Plant Health Assistant", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(images_router)


def _require_case(case_id: str) -> dict:
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(404, "case not found")
    return case


def _queue_item(case: dict) -> dict:
    last = (case["turns"] or [{}])[-1] if case["turns"] else {}
    assessment = last.get("assessment") or {}
    disputed = any(f.get("disputed") for f in case.get("feedback") or [])
    reviewed = bool(case.get("reviews"))
    outcome = assessment.get("outcome")
    coverage = assessment.get("coverage")
    score = 0
    reasons = []
    if disputed:
        score += QUEUE_WEIGHT_DISPUTE
        reasons.append("user_dispute")
    if case.get("review_requested"):
        score += QUEUE_WEIGHT_REQUESTED
        reasons.append("review_requested")
    if coverage == "none" or outcome in {"no_match", "unknown_cause", "conflict"}:
        score += QUEUE_WEIGHT_MISSING_KNOWLEDGE
        reasons.append("missing_or_weak_knowledge")
    if outcome in {"insufficient_image"}:
        score += QUEUE_WEIGHT_IMAGE_QUALITY
        reasons.append("image_quality")
    if not reviewed:
        score += QUEUE_WEIGHT_UNREVIEWED
        reasons.append("unreviewed")
    return {
        "case_id": case["case_id"],
        "created_at": case["created_at"],
        "crop": (case.get("meta") or {}).get("crop"),
        "priority": score,
        "reasons": reasons,
        "last_outcome": outcome,
        "coverage": coverage,
        "disputed": disputed,
        "reviewed": reviewed,
        "review_requested": case.get("review_requested"),
        "turn_count": len(case.get("turns") or []),
        "summary": case.get("summary") or {},
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "vlm_provider": VLM_PROVIDER,
        "approved_records": store.approved_count(),
        "supported_crops": list(SUPPORTED_CROPS),
        "role": "decision_support",
    }


@app.get("/config")
def get_config():
    """Public tunables for the UI. Change values in app/config.py (or env), not in the frontend."""
    return public_settings()


@app.get("/guidance/photos")
def photo_capture_guidance():
    return {"views": capture_guidance()}


@app.post("/cases")
def create_case(body: CaseCreate):
    return store.create_case(body.model_dump(exclude_none=True))


@app.get("/cases")
def list_cases():
    return [_queue_item(store.get_case(cid)) for cid in store.list_case_ids()]


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    return _require_case(case_id)


@app.patch("/cases/{case_id}")
def patch_case(case_id: str, body: CasePatch):
    _require_case(case_id)
    return store.update_case_meta(case_id, body.model_dump(exclude_none=True))


@app.post("/cases/{case_id}/messages")
async def add_message(case_id: str, text: str = Form(""), files: list[UploadFile] = File(default=[])):
    case = _require_case(case_id)
    text = text.strip()
    if not text and not files:
        raise HTTPException(400, "send a question, one or more images, or both")
    if len(files) > MAX_IMAGES_PER_MESSAGE:
        raise HTTPException(400, f"max {MAX_IMAGES_PER_MESSAGE} images per message")

    # Validate every file before saving any, so a rejected request leaves nothing behind.
    validated = []
    for f in files:
        try:
            data, issues = validate_image(f)
        except UploadError as e:
            raise HTTPException(e.status, e.message)
        validated.append((f.filename or "image", data, issues))

    case_dir = UPLOAD_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    image_paths, quality = [], []
    for name, data, issues in validated:
        # validate_image re-encodes without EXIF as JPEG; keep a stable stem for debugging.
        stem = Path(name).stem or "image"
        path = case_dir / f"{uuid.uuid4().hex}_{stem}.jpg"
        path.write_bytes(data)
        image_paths.append(str(path))
        quality.append(issues)

    assessment = run_assessment(case, text, image_paths, quality)
    turn_index = len(case["turns"])
    assessment["turn_index"] = turn_index
    store.add_turn(case_id, {"text": text, "images": image_paths, "assessment": assessment})
    if assessment.get("case_summary"):
        store.set_summary(case_id, assessment["case_summary"])
    return assessment


@app.post("/cases/{case_id}/request-review")
def request_review(case_id: str):
    _require_case(case_id)
    store.flag_review(case_id)
    return {"case_id": case_id, "review_requested": True}


def _check_turn_index(case: dict, turn_index: int | None) -> None:
    if turn_index is not None and not 0 <= turn_index < len(case["turns"]):
        raise HTTPException(400, f"turn_index {turn_index} does not exist in this case")


@app.post("/cases/{case_id}/feedback")
def add_feedback(case_id: str, body: FeedbackIn):
    case = _require_case(case_id)
    _check_turn_index(case, body.turn_index)
    return store.add_feedback(case_id, body.model_dump())


@app.post("/cases/{case_id}/reviews")
def add_review(case_id: str, body: ReviewIn):
    case = _require_case(case_id)
    _check_turn_index(case, body.turn_index)
    return store.add_review(case_id, body.model_dump())


@app.get("/uploads/{case_id}/{name}")
def get_upload(case_id: str, name: str):
    root = (UPLOAD_DIR / case_id).resolve()
    target = (root / name).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(404, "image not found")
    if not target.is_file() or target.suffix.lower() not in IMAGE_ALLOWED_EXT:
        raise HTTPException(404, "image not found")
    return FileResponse(target)


@app.get("/reviews/queue")
def review_queue():
    items = [_queue_item(store.get_case(cid)) for cid in store.list_case_ids()]
    items.sort(key=lambda x: (-x["priority"], x["created_at"]))
    return {"items": items}


@app.get("/knowledge/records")
def list_records():
    return {"records": store.all_records(), "sources": store.list_sources()}


@app.patch("/knowledge/records/{record_id}")
def patch_record(record_id: str, body: RecordPatch):
    rec = store.update_record(record_id, body.model_dump(exclude_none=True))
    if not rec:
        raise HTTPException(404, "record not found")
    return rec


@app.get("/knowledge/coverage")
def coverage():
    return {"register": store.coverage_register()}


@app.get("/knowledge/gaps")
def gaps():
    return {"items": store.list_gaps()}


@app.get("/evidence/{evidence_id:path}")
def get_evidence(evidence_id: str):
    rec = store.get_record(evidence_id)
    if rec:
        src = store.get_source(rec.get("source_id")) if rec.get("source_id") else None
        return {
            "id": rec["record_id"],
            "type": "text",
            "record": rec,
            "source": src,
            "label_basis": rec.get("label_basis"),
            "review_status": rec.get("review_status"),
        }
    for rec in store.all_records():
        for img in rec.get("images") or []:
            if img.get("image_path") == evidence_id:
                src = store.get_source(rec.get("source_id")) if rec.get("source_id") else None
                return {
                    "id": evidence_id,
                    "type": "image",
                    "record": rec,
                    "image": img,
                    "source": src,
                    "label_basis": img.get("label_status") or rec.get("label_basis"),
                    "review_status": rec.get("review_status"),
                }
    raise HTTPException(404, "evidence not found")


@app.get("/dashboard")
def dashboard():
    return store.dashboard()


@app.post("/eval/run")
def eval_run():
    return run_eval()
