"""SQLite storage: cases, turns, sources, feedback, reviews, knowledge gaps.

Approved records are loaded from SEED_FILE and sources from SOURCES_FILE at startup.
Restart the server after replacing the seed file; in-app edits are written back.
"""
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from app.config import DB_PATH, GAPS_FILE, SEED_FILE, SOURCES_FILE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_record(rec: dict) -> dict:
    rec = dict(rec)
    rec.setdefault("condition_id", (rec.get("condition") or "").lower().replace(" ", "_"))
    rec.setdefault("publication_status", rec.get("review_status"))
    rec.setdefault("category", "disease")
    rec.setdefault("synonyms", [])
    rec.setdefault("affected_organs", [])
    rec.setdefault("progression", None)
    rec.setdefault("context", None)
    rec.setdefault("version", "1")
    rec.setdefault("label_basis", "expert_confirmed" if rec.get("review_status") == "approved" else "expert_suspected")
    rec.setdefault("images", [])
    return rec


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                meta TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL REFERENCES cases(case_id),
                text TEXT,
                images TEXT,
                assessment TEXT,
                at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                title TEXT,
                publisher TEXT,
                url TEXT,
                published_on TEXT,
                licence TEXT,
                reviewer TEXT,
                version TEXT,
                scope TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                turn_index INTEGER,
                helpful INTEGER NOT NULL,
                usefulness INTEGER,
                disputed INTEGER NOT NULL,
                comment TEXT,
                at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                turn_index INTEGER,
                reviewer TEXT NOT NULL,
                decision TEXT NOT NULL,
                corrected_condition TEXT,
                notes TEXT,
                at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT,
                crop TEXT,
                outcome TEXT,
                question TEXT,
                reason TEXT,
                at TEXT NOT NULL
            );
        """)
        self._migrate()
        self._records: dict[str, dict] = {}
        self._load_seed()
        self._load_sources()

    def _migrate(self) -> None:
        def columns(table: str) -> set[str]:
            return {r[1] for r in self._db.execute(f"PRAGMA table_info({table})")}

        case_cols = columns("cases")
        if "summary" not in case_cols:
            self._db.execute("ALTER TABLE cases ADD COLUMN summary TEXT")
        if "review_requested" not in case_cols:
            self._db.execute("ALTER TABLE cases ADD COLUMN review_requested INTEGER NOT NULL DEFAULT 0")

        fb_cols = columns("feedback")
        if "next_action_helped" not in fb_cols:
            self._db.execute("ALTER TABLE feedback ADD COLUMN next_action_helped INTEGER")

        rv_cols = columns("reviews")
        for col, spec in (
            ("confirmation_method", "TEXT"),
            ("confidence_basis", "TEXT"),
            ("publish_decision", "TEXT"),
        ):
            if col not in rv_cols:
                self._db.execute(f"ALTER TABLE reviews ADD COLUMN {col} {spec}")
        self._db.commit()

    # ---- knowledge ----
    def _load_seed(self):
        if not SEED_FILE.exists():
            return
        for rec in json.loads(SEED_FILE.read_text(encoding="utf-8")):
            rec = _normalise_record(rec)
            self._records[rec["record_id"]] = rec

    def _persist_records(self) -> None:
        SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
        SEED_FILE.write_text(
            json.dumps(list(self._records.values()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_sources(self):
        if not SOURCES_FILE.exists():
            return
        rows = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
        with self._lock:
            for s in rows:
                self._db.execute(
                    "INSERT OR REPLACE INTO sources (source_id, title, publisher, url, published_on, "
                    "licence, reviewer, version, scope) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (s["source_id"], s.get("title"), s.get("publisher"), s.get("url"),
                     s.get("published_on"), s.get("licence"), s.get("reviewer"),
                     s.get("version"), s.get("scope")),
                )
            self._db.commit()

    def get_source(self, source_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        return dict(row) if row else None

    def list_sources(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM sources ORDER BY source_id").fetchall()
        return [dict(r) for r in rows]

    def all_records(self) -> list[dict]:
        return list(self._records.values())

    def approved_records(self) -> list[dict]:
        return [
            r for r in self._records.values()
            if r.get("review_status") == "approved" and r.get("publication_status") != "retired"
        ]

    def approved_count(self) -> int:
        return len(self.approved_records())

    def get_record(self, record_id: str) -> dict | None:
        return self._records.get(record_id)

    def update_record(self, record_id: str, patch: dict) -> dict | None:
        rec = self._records.get(record_id)
        if not rec:
            return None
        allowed = {
            "review_status", "reviewed_by", "symptoms", "distinguishing_features",
            "management_general", "limitations", "next_step", "alternatives",
        }
        for k, v in patch.items():
            if k in allowed and v is not None:
                rec[k] = v
        if rec.get("review_status") == "retired":
            rec["publication_status"] = "retired"
        elif rec.get("review_status") == "approved":
            rec["publication_status"] = "approved"
            rec["label_basis"] = rec.get("label_basis") or "expert_confirmed"
        else:
            rec["publication_status"] = rec.get("review_status")
        rec["reviewed_on"] = rec.get("reviewed_on") or _now()[:10]
        self._records[record_id] = rec
        with self._lock:
            self._persist_records()
        return rec

    def coverage_register(self) -> list[dict]:
        rows = []
        for rec in self._records.values():
            imgs = rec.get("images") or []
            rows.append({
                "crop": rec.get("crop"),
                "condition": rec.get("condition"),
                "category": rec.get("category"),
                "review_status": rec.get("review_status"),
                "publication_status": rec.get("publication_status"),
                "has_symptoms": bool(rec.get("symptoms") and "TODO" not in str(rec.get("symptoms")).upper()),
                "has_management": bool(rec.get("management_general") and "TODO" not in str(rec.get("management_general")).upper()),
                "image_count": len(imgs),
                "source_id": rec.get("source_id"),
                "label_basis": rec.get("label_basis"),
                "weak_in_evaluation": bool(rec.get("weak_in_evaluation")),
            })
        return rows

    # ---- cases ----
    def create_case(self, meta: dict) -> dict:
        case_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._lock:
            self._db.execute(
                "INSERT INTO cases (case_id, meta, created_at, summary, review_requested) VALUES (?, ?, ?, ?, 0)",
                (case_id, json.dumps(meta), now, json.dumps({})),
            )
            self._db.commit()
        return {
            "case_id": case_id, "meta": meta, "created_at": now,
            "turns": [], "feedback": [], "reviews": [],
            "summary": {}, "review_requested": False,
        }

    def update_case_meta(self, case_id: str, patch: dict) -> dict | None:
        case = self.get_case(case_id)
        if not case:
            return None
        meta = {**case["meta"], **{k: v for k, v in patch.items() if v is not None}}
        with self._lock:
            self._db.execute("UPDATE cases SET meta = ? WHERE case_id = ?", (json.dumps(meta), case_id))
            self._db.commit()
        case["meta"] = meta
        return case

    def set_summary(self, case_id: str, summary: dict) -> None:
        with self._lock:
            self._db.execute("UPDATE cases SET summary = ? WHERE case_id = ?", (json.dumps(summary), case_id))
            self._db.commit()

    def flag_review(self, case_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE cases SET review_requested = 1 WHERE case_id = ?", (case_id,))
            self._db.commit()

    def get_case(self, case_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                return None
            turns = self._db.execute(
                "SELECT text, images, assessment, at FROM turns WHERE case_id = ? ORDER BY id", (case_id,)
            ).fetchall()
            fb = self._db.execute(
                "SELECT turn_index, helpful, usefulness, disputed, next_action_helped, comment, at FROM feedback "
                "WHERE case_id = ? ORDER BY id", (case_id,)).fetchall()
            rv = self._db.execute(
                "SELECT turn_index, reviewer, decision, confirmation_method, confidence_basis, "
                "corrected_condition, notes, publish_decision, at FROM reviews "
                "WHERE case_id = ? ORDER BY id", (case_id,)).fetchall()
        summary = {}
        if row["summary"]:
            try:
                summary = json.loads(row["summary"])
            except json.JSONDecodeError:
                summary = {}
        return {
            "case_id": row["case_id"],
            "meta": json.loads(row["meta"]),
            "created_at": row["created_at"],
            "summary": summary,
            "review_requested": bool(row["review_requested"]) if "review_requested" in row.keys() else False,
            "turns": [{
                "text": t["text"],
                "images": json.loads(t["images"]),
                "assessment": json.loads(t["assessment"]),
                "at": t["at"],
            } for t in turns],
            "feedback": [{
                "turn_index": f["turn_index"], "helpful": bool(f["helpful"]),
                "usefulness": f["usefulness"], "disputed": bool(f["disputed"]),
                "next_action_helped": None if f["next_action_helped"] is None else bool(f["next_action_helped"]),
                "comment": f["comment"], "at": f["at"],
            } for f in fb],
            "reviews": [{k: r[k] for k in r.keys()} for r in rv],
        }

    def add_turn(self, case_id: str, turn: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO turns (case_id, text, images, assessment, at) VALUES (?, ?, ?, ?, ?)",
                (case_id, turn.get("text", ""), json.dumps(turn.get("images", [])),
                 json.dumps(turn.get("assessment")), _now()),
            )
            self._db.commit()

    def add_feedback(self, case_id: str, fb: dict) -> dict:
        at = _now()
        helped = fb.get("next_action_helped")
        with self._lock:
            self._db.execute(
                "INSERT INTO feedback (case_id, turn_index, helpful, usefulness, disputed, "
                "next_action_helped, comment, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (case_id, fb.get("turn_index"), int(fb["helpful"]), fb.get("usefulness"),
                 int(fb.get("disputed", False)),
                 None if helped is None else int(helped),
                 fb.get("comment"), at),
            )
            self._db.commit()
        return {**fb, "at": at}

    def add_review(self, case_id: str, rv: dict) -> dict:
        at = _now()
        with self._lock:
            self._db.execute(
                "INSERT INTO reviews (case_id, turn_index, reviewer, decision, confirmation_method, "
                "confidence_basis, corrected_condition, notes, publish_decision, at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (case_id, rv.get("turn_index"), rv["reviewer"], rv["decision"],
                 rv.get("confirmation_method") or "none", rv.get("confidence_basis"),
                 rv.get("corrected_condition"), rv.get("notes"),
                 rv.get("publish_decision") or "exclude", at),
            )
            self._db.commit()
        return {**rv, "at": at}

    def log_gap(self, item: dict) -> None:
        at = _now()
        with self._lock:
            self._db.execute(
                "INSERT INTO knowledge_gaps (case_id, crop, outcome, question, reason, at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (item.get("case_id"), item.get("crop"), item.get("outcome"),
                 item.get("question"), item.get("reason"), at),
            )
            self._db.commit()
        GAPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with GAPS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**item, "at": at}, ensure_ascii=False) + "\n")

    def list_gaps(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT case_id, crop, outcome, question, reason, at FROM knowledge_gaps "
                "ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_case_ids(self) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT case_id FROM cases ORDER BY created_at DESC").fetchall()
        return [r["case_id"] for r in rows]

    def dashboard(self) -> dict:
        with self._lock:
            n_cases = self._db.execute("SELECT COUNT(*) AS n FROM cases").fetchone()["n"]
            n_turns = self._db.execute("SELECT COUNT(*) AS n FROM turns").fetchone()["n"]
            n_reviews = self._db.execute("SELECT COUNT(*) AS n FROM reviews").fetchone()["n"]
            n_feedback = self._db.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
            n_disputed = self._db.execute(
                "SELECT COUNT(*) AS n FROM feedback WHERE disputed = 1").fetchone()["n"]
            n_gaps = self._db.execute("SELECT COUNT(*) AS n FROM knowledge_gaps").fetchone()["n"]
            n_review_req = self._db.execute(
                "SELECT COUNT(*) AS n FROM cases WHERE review_requested = 1").fetchone()["n"]
            assessments = self._db.execute("SELECT assessment FROM turns ORDER BY id DESC LIMIT 200").fetchall()
        outcomes: dict[str, int] = {}
        latencies: list[int] = []
        coverage: dict[str, int] = {}
        policy_hits = 0
        for row in assessments:
            try:
                a = json.loads(row["assessment"])
            except json.JSONDecodeError:
                continue
            outcomes[a.get("outcome") or "unknown"] = outcomes.get(a.get("outcome") or "unknown", 0) + 1
            coverage[a.get("coverage") or "none"] = coverage.get(a.get("coverage") or "none", 0) + 1
            total = (a.get("latency_ms") or {}).get("total")
            if isinstance(total, (int, float)):
                latencies.append(int(total))
            policy_hits += int(a.get("policy_removals") or 0)
        latencies.sort()
        p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else None
        median = latencies[len(latencies) // 2] if latencies else None
        return {
            "cases": n_cases,
            "turns": n_turns,
            "reviews": n_reviews,
            "feedback": n_feedback,
            "disputed": n_disputed,
            "knowledge_gaps": n_gaps,
            "review_requested": n_review_req,
            "approved_records": self.approved_count(),
            "pending_records": sum(1 for r in self._records.values() if r.get("review_status") == "pending"),
            "outcomes": outcomes,
            "coverage": coverage,
            "policy_removals": policy_hits,
            "latency_ms": {"median": median, "p95": p95, "n": len(latencies)},
        }


store = Store()
