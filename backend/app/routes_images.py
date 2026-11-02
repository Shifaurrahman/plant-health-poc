"""Serve dataset photos so the frontend can show cited reference images.

Only files under configured dataset prefixes, with image extensions, are served.
Prefixes and extensions come from config.py.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import DATASET_ROOT, IMAGE_ALLOWED_EXT, IMAGE_ALLOWED_PREFIXES

router = APIRouter()


@router.get("/images/{rel_path:path}")
def get_image(rel_path: str):
    if not rel_path.startswith(IMAGE_ALLOWED_PREFIXES):
        raise HTTPException(404, "image not found")
    root = DATASET_ROOT.resolve()
    target = (root / rel_path).resolve()
    # Blocks ../ escapes: the resolved file must still sit under the dataset root.
    if root not in target.parents or not target.is_file() or target.suffix.lower() not in IMAGE_ALLOWED_EXT:
        raise HTTPException(404, "image not found")
    return FileResponse(target)
