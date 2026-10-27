"""Serve dataset photos so the frontend can show cited reference images.

Only files under the dataset's Raw / Phase folders, with image extensions, are served.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
DATASET_ROOT = BASE_DIR / "data" / "raw" / "sri_lanka_brinjal"
ALLOWED_PREFIXES = ("Raw_Dataset/", "Phase_I_Dataset/", "Phase_II_Dataset/")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

router = APIRouter()


@router.get("/images/{rel_path:path}")
def get_image(rel_path: str):
    if not rel_path.startswith(ALLOWED_PREFIXES):
        raise HTTPException(404, "image not found")
    root = DATASET_ROOT.resolve()
    target = (root / rel_path).resolve()
    # Blocks ../ escapes: the resolved file must still sit under the dataset root.
    if root not in target.parents or not target.is_file() or target.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(404, "image not found")
    return FileResponse(target)