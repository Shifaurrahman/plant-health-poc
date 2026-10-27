import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("PH_DATA_DIR", BASE_DIR / "data"))
SEED_FILE = DATA_DIR / "seed" / "records.json"   # copy records.example.json here and fill in
UPLOAD_DIR = DATA_DIR / "uploads"
VLM_PROVIDER = os.getenv("PH_VLM_PROVIDER", "stub")
MAX_IMAGES_PER_MESSAGE = 4
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
