import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_IMAGE_TYPES, MAX_IMAGES_PER_MESSAGE, UPLOAD_DIR, VLM_PROVIDER
from app.schemas import CaseCreate
from app.services.pipeline import run_assessment
from app.store import store
from app.routes_images import router as images_router

app = FastAPI(title="Plant Health POC", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(images_router)


@app.get("/health")
def health():
    return {"status": "ok", "vlm_provider": VLM_PROVIDER, "approved_records": store.approved_count()}


@app.post("/cases")
def create_case(body: CaseCreate):
    return store.create_case(body.model_dump(exclude_none=True))


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(404, "case not found")
    return case


@app.post("/cases/{case_id}/messages")
async def add_message(case_id: str, text: str = Form(""), files: list[UploadFile] = File(default=[])):
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(404, "case not found")
    if not text.strip() and not files:
        raise HTTPException(400, "send a question, one or more images, or both")
    if len(files) > MAX_IMAGES_PER_MESSAGE:
        raise HTTPException(400, f"max {MAX_IMAGES_PER_MESSAGE} images per message")

    case_dir = UPLOAD_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for f in files:
        if f.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, f"unsupported file type: {f.content_type}")
        path = case_dir / f"{uuid.uuid4().hex}_{Path(f.filename or 'image').name}"
        with path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        image_paths.append(str(path))

    assessment = run_assessment(case, text.strip(), image_paths)
    store.add_turn(case_id, {"text": text.strip(), "images": image_paths, "assessment": assessment})
    return assessment
