"""Version fingerprints stored with every assessment, so an answer can be traced later."""
import hashlib
from pathlib import Path

from app.config import (
    APP_VERSION, EMBED_CACHE, INDEX_CSV, KNOWLEDGE_VERSION, POLICY_VERSION, SEED_FILE, SOURCES_FILE,
)
from app.services.embedder import MODEL

_cache: dict[tuple[str, int], str] = {}


def _sha(path: Path) -> str | None:
    """Short SHA-256 of a file. Re-hashed only when the file's modification time changes."""
    if not path.exists():
        return None
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _cache:
        _cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return _cache[key]


def current_versions(vlm_name: str, approved_count: int) -> dict:
    return {
        "app": APP_VERSION,
        "policy": POLICY_VERSION,
        "model": vlm_name,
        "prompt": "none",
        "embedding": MODEL,
        "index_sha": _sha(INDEX_CSV),
        "embedding_cache_sha": _sha(EMBED_CACHE),
        "records_sha": _sha(SEED_FILE),
        "sources_sha": _sha(SOURCES_FILE),
        "knowledge": KNOWLEDGE_VERSION,
        "approved_records": approved_count,
    }