"""Central settings. Change values here (or via env vars) — do not scatter magic numbers in services.

Environment overrides (optional):
  PH_DATA_DIR, PH_VLM_PROVIDER, PH_VLM_MODEL, PH_VLM_TIMEOUT_S,
  PH_VLM_ALLOW_REAL_PHOTOS, PH_CORS_ORIGINS, PH_API_PUBLIC_URL,
  PH_SUPPORTED_CROPS, PH_DEFAULT_REGION, PH_HEALTHY_CLASS,
  PH_CHANNEL_K, PH_EVIDENCE_K, PH_TOP_K, PH_MAJORITY_SHARE,
  PH_MAX_IMAGES, PH_MAX_UPLOAD_MB,
  PH_EMBED_MODEL, PH_EMBED_DTYPE, PH_EMBED_MAX_PIXELS, PH_EMBED_BATCH,
  PH_EMBED_QUERY_INSTRUCTION, PH_QWEN_REPO_DIR, PH_EMBED_CACHE_NAME
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/

# .env wins over shell variables, so one file controls the settings.
load_dotenv(BASE_DIR / ".env", override=True)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(x.strip() for x in raw.split(",") if x.strip())


# ---- embeddings (Qwen3-VL-Embedding: text and images share one vector space) ----
EMBED_MODEL = _env("PH_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-2B")  # Hugging Face id or local folder
EMBED_DTYPE = _env("PH_EMBED_DTYPE", "bfloat16")                    # bfloat16 | float16 | float32
EMBED_MAX_PIXELS = _env_int("PH_EMBED_MAX_PIXELS", 0)               # 0 = model default; lower = faster
EMBED_BATCH = _env_int("PH_EMBED_BATCH", 8)                         # images per batch, offline cache build
EMBED_QUERY_INSTRUCTION = _env(
    "PH_EMBED_QUERY_INSTRUCTION",
    "Retrieve plant disease reference records relevant to the query.",
)
# The Qwen3VLEmbedder class lives in github.com/QwenLM/Qwen3-VL-Embedding (not on pip).
# Clone it and point this at the folder that contains its src/ directory.
QWEN_REPO_DIR = _env("PH_QWEN_REPO_DIR", "")

# ---- paths ----
DATA_DIR = Path(_env("PH_DATA_DIR", str(BASE_DIR / "data")))

SEED_FILE = DATA_DIR / "seed" / "records.json"
SOURCES_FILE = DATA_DIR / "seed" / "sources.json"
DB_PATH = DATA_DIR / "poc.sqlite3"
UPLOAD_DIR = DATA_DIR / "uploads"
GAPS_FILE = DATA_DIR / "logs" / "knowledge_gaps.jsonl"
EVAL_FILE = DATA_DIR / "eval" / "cases.json"

DATASET_ROOT = DATA_DIR / "raw" / _env("PH_DATASET_FOLDER", "sri_lanka_brinjal")
INDEX_CSV = DATA_DIR / "splits" / _env("PH_INDEX_CSV_NAME", "main_index.csv")
# One cache file per model: vectors from different models must never be mixed.
EMBED_CACHE = DATA_DIR / "embeddings" / _env(
    "PH_EMBED_CACHE_NAME", Path(EMBED_MODEL).name.lower() + ".npz")

# ---- pilot scope (change crops/region here) ----
SUPPORTED_CROPS = tuple(c.lower() for c in _env_csv("PH_SUPPORTED_CROPS", ("brinjal",)))
CROP = SUPPORTED_CROPS[0] if SUPPORTED_CROPS else "brinjal"
DEFAULT_REGION = _env("PH_DEFAULT_REGION", "Sri Lanka")
HEALTHY_CLASS = _env("PH_HEALTHY_CLASS", "Healthy")
DEFAULT_PLANT_PART = _env("PH_DEFAULT_PLANT_PART", "leaf")

# ---- retrieval (tuning knobs, not diagnostic thresholds) ----
CHANNEL_K = _env_int("PH_CHANNEL_K", 15)
EVIDENCE_K = _env_int("PH_EVIDENCE_K", 8)
TOP_K = _env_int("PH_TOP_K", 10)
MAJORITY_SHARE = _env_float("PH_MAJORITY_SHARE", 0.6)
MAX_RETRIEVAL_ROUNDS = _env_int("PH_MAX_RETRIEVAL_ROUNDS", 1)
# ---- match thresholds (calibrate on held-out photos; defaults keep the old behaviour) ----
PHOTO_MIN_SIMILARITY = _env_float("PH_PHOTO_MIN_SIMILARITY", 0.0)  # photo matches below this don't vote
PHOTO_MIN_VOTES = _env_int("PH_PHOTO_MIN_VOTES", 3)                # fewer surviving matches -> unknown_cause
SEMANTIC_MIN_SCORE = _env_float("PH_SEMANTIC_MIN_SCORE", 0.0)      # text and image-to-text hits below this are dropped

# ---- text relevance (model decides which shortlisted records match the farmer's words) ----
RELEVANCE_PROVIDER = _env("PH_RELEVANCE_PROVIDER", "none").lower()   # none | anthropic
RELEVANCE_MODEL = _env("PH_RELEVANCE_MODEL", "claude-haiku-5-5")
RELEVANCE_SHORTLIST_K = _env_int("PH_RELEVANCE_SHORTLIST_K", 8)
RELEVANCE_TIMEOUT_S = _env_float("PH_RELEVANCE_TIMEOUT_S", 20)

# ---- answer shape ----
MAX_CANDIDATE_CAUSES = _env_int("PH_MAX_CANDIDATE_CAUSES", 3)
MAX_ALTERNATIVES = _env_int("PH_MAX_ALTERNATIVES", 3)
MAX_QUESTIONS = _env_int("PH_MAX_QUESTIONS", 3)
MAX_OBSERVATIONS = _env_int("PH_MAX_OBSERVATIONS", 6)
MAX_GENERAL_KNOWLEDGE = _env_int("PH_MAX_GENERAL_KNOWLEDGE", 3)

# ---- uploads ----
MAX_IMAGES_PER_MESSAGE = _env_int("PH_MAX_IMAGES", 4)
MAX_UPLOAD_MB = _env_int("PH_MAX_UPLOAD_MB", 8)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MIN_IMAGE_SIDE_PX = _env_int("PH_MIN_IMAGE_SIDE_PX", 224)
MIN_PIXEL_STD = _env_float("PH_MIN_PIXEL_STD", 5.0)
UPLOAD_JPEG_QUALITY = _env_int("PH_UPLOAD_JPEG_QUALITY", 90)

# ---- VLM ----
VLM_PROVIDER = _env("PH_VLM_PROVIDER", "stub").lower()
VLM_TIMEOUT_S = _env_float("PH_VLM_TIMEOUT_S", 30)
VLM_MAX_SIDE_PX = _env_int("PH_VLM_MAX_SIDE_PX", 1024)
VLM_MAX_TOKENS = _env_int("PH_VLM_MAX_TOKENS", 1200)
VLM_JPEG_QUALITY = _env_int("PH_VLM_JPEG_QUALITY", 85)

# ---- HTTP / CORS ----
CORS_ORIGINS = list(_env_csv("PH_CORS_ORIGINS", ("http://localhost:5173",)))
API_PUBLIC_URL = _env("PH_API_PUBLIC_URL", "http://127.0.0.1:8000")

# ---- dataset image serving ----
IMAGE_ALLOWED_PREFIXES = tuple(_env_csv(
    "PH_IMAGE_ALLOWED_PREFIXES",
    ("Raw_Dataset/", "Phase_I_Dataset/", "Phase_II_Dataset/"),
))
IMAGE_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# ---- expert queue priority weights ----
QUEUE_WEIGHT_DISPUTE = _env_int("PH_QUEUE_WEIGHT_DISPUTE", 4)
QUEUE_WEIGHT_REQUESTED = _env_int("PH_QUEUE_WEIGHT_REQUESTED", 3)
QUEUE_WEIGHT_MISSING_KNOWLEDGE = _env_int("PH_QUEUE_WEIGHT_MISSING_KNOWLEDGE", 3)
QUEUE_WEIGHT_IMAGE_QUALITY = _env_int("PH_QUEUE_WEIGHT_IMAGE_QUALITY", 1)
QUEUE_WEIGHT_UNREVIEWED = _env_int("PH_QUEUE_WEIGHT_UNREVIEWED", 1)

# ---- versions ----
APP_VERSION = "1.0.0"
POLICY_VERSION = "1.0"
KNOWLEDGE_VERSION = "seed-1"


def crops_label() -> str:
    """Human-readable supported crop list for messages."""
    if not SUPPORTED_CROPS:
        return "no crops"
    if len(SUPPORTED_CROPS) == 1:
        return SUPPORTED_CROPS[0]
    return ", ".join(SUPPORTED_CROPS[:-1]) + f" and {SUPPORTED_CROPS[-1]}"


def public_settings() -> dict:
    """Safe settings the frontend may load so it does not hardcode pilot values."""
    return {
        "api_public_url": API_PUBLIC_URL,
        "app_version": APP_VERSION,
        "supported_crops": list(SUPPORTED_CROPS),
        "default_crop": CROP,
        "default_region": DEFAULT_REGION,
        "default_plant_part": DEFAULT_PLANT_PART,
        "healthy_class": HEALTHY_CLASS,
        "max_images_per_message": MAX_IMAGES_PER_MESSAGE,
        "max_upload_mb": MAX_UPLOAD_MB,
        "allowed_image_types": sorted(ALLOWED_IMAGE_TYPES),
        "vlm_provider": VLM_PROVIDER,
        "channel_k": CHANNEL_K,
        "evidence_k": EVIDENCE_K,
        "top_k": TOP_K,
        "majority_share": MAJORITY_SHARE,
        "max_candidate_causes": MAX_CANDIDATE_CAUSES,
        "max_questions": MAX_QUESTIONS,
        "role": "decision_support",
    }