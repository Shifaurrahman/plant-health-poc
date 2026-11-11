from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import DATA_DIR

router = APIRouter()
RAW_ROOT = (DATA_DIR / "raw").resolve()
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _relative_raw_path(stored: str) -> str | None:
    """Turn any stored photo path (Windows, Colab, or relative) into a path under data/raw."""
    norm = unquote(stored).replace("\\", "/")
    key = "/data/raw/"
    i = norm.lower().find(key)
    if i >= 0:
        return norm[i + len(key):]
    if norm.lower().startswith("raw/"):
        return norm[4:]
    return norm


@router.get("/images/{stored:path}")
def get_reference_image(stored: str):
    rel = _relative_raw_path(stored)
    if not rel:
        raise HTTPException(404, "image not found")
    target = (RAW_ROOT / rel).resolve()
    if RAW_ROOT not in target.parents or target.suffix.lower() not in IMAGE_EXT or not target.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(target)