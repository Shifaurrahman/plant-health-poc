import json
import threading
import uuid
from datetime import datetime, timezone

from app.config import SEED_FILE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """In-memory case store (POC). Cases are lost on restart; records are loaded from the seed file."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cases: dict[str, dict] = {}
        self._records: dict[str, dict] = {}
        self._load_seed()

    def _load_seed(self):
        if not SEED_FILE.exists():
            return
        for rec in json.loads(SEED_FILE.read_text(encoding="utf-8")):
            self._records[rec["record_id"]] = rec

    def approved_records(self) -> list[dict]:
        # Only expert-approved records may enter retrieval.
        return [r for r in self._records.values() if r.get("review_status") == "approved"]

    def approved_count(self) -> int:
        return len(self.approved_records())

    def create_case(self, meta: dict) -> dict:
        case = {"case_id": uuid.uuid4().hex[:12], "meta": meta, "created_at": _now(), "turns": []}
        with self._lock:
            self._cases[case["case_id"]] = case
        return case

    def get_case(self, case_id: str) -> dict | None:
        return self._cases.get(case_id)

    def add_turn(self, case_id: str, turn: dict):
        with self._lock:
            self._cases[case_id]["turns"].append({**turn, "at": _now()})


store = Store()
