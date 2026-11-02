"""Upload checks: size limit, real file type (magic bytes), decodable image, basic quality.

Hard failures raise UploadError (the request is rejected).
Quality problems are returned as issues: the photo is kept, but the answer says it is not suitable.
"""
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import (
    ALLOWED_IMAGE_TYPES, MAX_UPLOAD_BYTES, MIN_IMAGE_SIDE_PX, MIN_PIXEL_STD,
    UPLOAD_JPEG_QUALITY,
)


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def sniff_type(data: bytes) -> str | None:
    """Detect the real file type from the first bytes, not from the filename or declared type."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def quality_issues(img: Image.Image) -> list[str]:
    issues = []
    w, h = img.size
    if min(w, h) < MIN_IMAGE_SIDE_PX:
        issues.append(f"too small ({w}x{h} px; the short side needs at least {MIN_IMAGE_SIDE_PX} px)")
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    if gray.std() < MIN_PIXEL_STD:
        issues.append("almost uniform: no visible detail to compare")
    return issues


def validate_image(upload) -> tuple[bytes, list[str]]:
    """Return (validated bytes, quality issues). Raises UploadError on hard failures."""
    declared = upload.content_type
    if declared not in ALLOWED_IMAGE_TYPES:
        raise UploadError(400, f"unsupported file type: {declared}")

    # Read one byte past the limit so oversized files are detected without loading everything.
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(413, f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise UploadError(400, "file is empty")

    actual = sniff_type(data)
    if actual is None:
        raise UploadError(400, "file content is not a JPEG, PNG or WebP image")
    if actual != declared:
        raise UploadError(400, f"declared {declared} but the file content is {actual}")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise UploadError(400, "image cannot be decoded")

    issues = quality_issues(img)
    # Re-encode as JPEG without EXIF / GPS metadata.
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=UPLOAD_JPEG_QUALITY)
    return buf.getvalue(), issues