#!/usr/bin/env python3
"""Smoke test for the v1.0 plan-aligned features. Uses a fake embedder, so no model download is needed.

Run from backend/:  python tests/smoke_test.py
Needs:              pip install httpx fastapi pillow numpy
"""
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
DATA = TMP / "data"
os.environ["PH_DATA_DIR"] = str(DATA)
os.environ["PH_EMBED_CACHE_NAME"] = "test_embeddings.npz"  # .env must not redirect the test cache
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

failures = []


def check(name: str, ok: bool) -> None:
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        failures.append(name)


# ---------- fake data: index, embeddings, records, sources ----------
rng = np.random.default_rng(0)
CLASS_AXIS = {"Healthy": [0, 0, 1], "Little Leaf": [1, 0, 0], "Mosaic Virus": [0, 1, 0]}
rows, vecs = [], {}
for cls, base in CLASS_AXIS.items():
    folder = cls.replace(" ", "_")
    for i in range(6):
        rel = f"Raw_Dataset/{folder}/{folder[:2].lower()}_{i:03d}.jpg"
        v = np.array(base, float) + 0.01 * rng.standard_normal(3)
        rows.append((rel, folder))
        vecs[rel] = v / np.linalg.norm(v)

(DATA / "splits").mkdir(parents=True)
(DATA / "embeddings").mkdir(parents=True)
(DATA / "seed").mkdir(parents=True)
with open(DATA / "splits" / "main_index.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["relative_path", "class", "phase", "derivation", "source_image", "collection_site"])
    for rel, folder in rows:
        writer.writerow([rel, folder, "Raw", "original", Path(rel).name, "TEST"])
np.savez(DATA / "embeddings" / "test_embeddings.npz",
         paths=np.array(list(vecs)), feats=np.stack(list(vecs.values())))


def record(rid, cond, symptoms, weak=False, images=None):
    return {"record_id": rid, "crop": "brinjal", "condition": cond, "category": "disease",
            "synonyms": [], "symptoms": symptoms, "distinguishing_features": "test",
            "alternatives": [], "management_general": "Remove infected plants",
            "source_id": "src_test", "review_status": "approved", "reviewed_by": "dev-test",
            "reviewed_on": "2026-10-27", "weak_in_evaluation": weak,
            "evaluation_note": "Test note: weak." if weak else None,
            "limitations": ["Test record limitation."] if cond == "Little Leaf" else [],
            "next_step": None, "images": images or []}


(DATA / "seed" / "records.json").write_text(json.dumps([
    record("rec_h", "Healthy", "even green leaves"),
    record("rec_ll", "Little Leaf", "small narrow leaves",
           images=[{"image_path": "Raw_Dataset/Little_Leaf/ll_000.jpg", "organ": "leaf", "label_status": "expert_confirmed"}]),
    record("rec_m", "Mosaic Virus", "mottled yellow pattern on leaf", weak=True),
]), encoding="utf-8")
(DATA / "seed" / "sources.json").write_text(json.dumps([{
    "source_id": "src_test", "title": "Test source (not real)", "publisher": "dev",
    "url": None, "published_on": "2026-10-27", "licence": "n/a", "reviewer": "dev-test",
    "version": "0", "scope": "test"}]), encoding="utf-8")

# ---------- import the app and swap in the fake embedder ----------
import app.services.image_retrieval as ir  # noqa: E402
import app.services.retrieval as tr  # noqa: E402
from app.services.pipeline import decide_outcome  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
os.environ["PH_VLM_PROVIDER"] = "stub"  # tests never call the real API, even if .env sets anthropic
import app.services.pipeline as _pipeline  # noqa: E402
from app.services.vlm import StubVLM as _StubVLM  # noqa: E402
_pipeline.get_vlm = lambda: _StubVLM()  # force the stub regardless of .env
from fastapi.testclient import TestClient  # noqa: E402

QUERY = {"healthy": [0, 0, 1], "ll": [1, 0, 0], "mosaic": [0, 1, 0],
         "tiny": [1, 0, 0], "uniform": [1, 0, 0], "real": [1, 0, 0]}


def fake_embed(path):
    # Upload names are "{uuid}_{stem}.jpg"; map by stem.
    stem = Path(path).stem.split("_", 1)[-1]
    v = np.array(QUERY.get(stem, [1, 0, 0]), float)
    return v / np.linalg.norm(v)


def fake_encode_text(texts):
    # Deterministic unit vectors so smoke tests never load the real model.
    out = []
    for t in texts:
        key = "ll" if "little" in t.lower() or "narrow" in t.lower() else (
            "mosaic" if "mottle" in t.lower() else "healthy" if "green" in t.lower() else "ll"
        )
        v = np.array(QUERY[key], float)
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


ir.embed_image_file = fake_embed
tr.embed_image_file = fake_embed
tr.encode_query = fake_encode_text
tr.encode_document = fake_encode_text
client = TestClient(fastapi_app)


def png(size=(300, 300), uniform=False, jpeg=False):
    if uniform:
        arr = np.full((size[1], size[0], 3), 128, np.uint8)
    else:
        arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG" if jpeg else "PNG")
    return buf.getvalue()


def new_case(crop="brinjal"):
    return client.post("/cases", json={"crop": crop, "plant_part": "leaf"}).json()["case_id"]


def send(case_id, text="", files=()):
    return client.post(f"/cases/{case_id}/messages", data={"text": text},
                       files=[("files", (n, b, t)) for n, b, t in files])


# ---------- tests ----------
h = client.get("/health").json()
check("health reports app version and 3 approved records", h["app_version"] == "1.0.0" and h["approved_records"] == 3)
check("health names supported crops and decision-support role",
      h.get("supported_crops") == ["brinjal"] and h.get("role") == "decision_support")

case = new_case()
r = send(case, "brown spots", [("healthy.png", png(), "image/png")]).json()
check("healthy-looking photo -> outcome healthy_appearance", r["outcome"] == "healthy_appearance")
check("healthy outcome top candidate is Healthy", r["candidate_causes"][0]["name"] == "Healthy")
check("signals split relevance / coverage / confidence",
      {"retrieval_relevance", "evidence_coverage", "diagnostic_confidence"} <= set(r["signals"]))
check("diagnostic confidence is not assessed (value None)", r["signals"]["diagnostic_confidence"]["value"] is None)
check("candidate support has no similarity wording",
      all("similarity" not in s for c in r["candidate_causes"] for s in c["support"]))
check("healthy limitation added", any("does not rule out early symptoms" in l for l in r["limitations"]))
check("image evidence carries source, review status and reviewer",
      all(e["review_status"] == "approved" and e["reviewed_by"] == "dev-test"
          for e in r["retrieved_evidence"] if e["type"] == "image"))
check("evidence resolves source title from the sources table",
      any(e.get("source_title") == "Test source (not real)" for e in r["retrieved_evidence"]))
check("assessment logs latency", {"total", "image_match", "text_match"} <= set(r["latency_ms"]))
check("assessment logs versions", r["versions"]["index_sha"] and r["versions"]["records_sha"]
      and r["versions"]["embedding"].startswith("Qwen"))

r = send(case, "", [("ll.png", png(), "image/png")]).json()
check("Little Leaf photo -> outcome match", r["outcome"] == "match")
check("Little Leaf top candidate", r["candidate_causes"][0]["name"] == "Little Leaf")
check("record limitation shown for Little Leaf", "Test record limitation." in r["limitations"])

r = send(case, "", [("mosaic.png", png(), "image/png")]).json()
check("Mosaic photo carries its weak-evaluation note",
      any("Test note: weak" in a for c in r["candidate_causes"] for a in c["against_or_missing"]))

r = send(new_case("tea"), "", [("ll.png", png(), "image/png")]).json()
check("tea case -> outcome unsupported_crop, no candidates",
      r["outcome"] == "unsupported_crop" and r["candidate_causes"] == [])

r = send(case, "", [("tiny.png", png(size=(60, 60)), "image/png")]).json()
check("tiny photo -> outcome insufficient_image", r["outcome"] == "insufficient_image")
check("tiny photo reason is in limitations", any("too small" in l for l in r["limitations"]))

r = send(case, "", [("uniform.png", png(uniform=True), "image/png")]).json()
check("uniform photo -> outcome insufficient_image", r["outcome"] == "insufficient_image")

r = send(new_case(), "mottled leaves").json()
check("text-only keyword match -> outcome text_only_match", r["outcome"] == "text_only_match")
check("text-only questions skip image assessment", r["route"] == "text_qa" and r["observations"] == [])

r = send(case, "have the symptoms changed?").json()
check("follow-up without a new photo reuses earlier photos", r["route"] == "followup")

r = send(new_case(), "brown spots on leaves").json()
check("no words match -> outcome no_match", r["outcome"] == "no_match")

check("unknown_cause when neighbours split across classes",
      decide_outcome(True, ["x.png"], [[]],
                     [{"class": "Little Leaf"}, {"class": "Mosaic Virus"}, {"class": "Healthy"},
                      {"class": "Little Leaf"}, {"class": "Mosaic Virus"}], []) == "unknown_cause")

# ---------- upload validation ----------
c = new_case()
resp = send(c, "", [("fake.jpg", b"hello world" * 100, "image/jpeg")])
check("text renamed as .jpg is rejected (400)", resp.status_code == 400)
resp = send(c, "", [("big.jpg", b"\xff\xd8\xff" + b"0" * (9 * 1024 * 1024), "image/jpeg")])
check("file over 8 MB is rejected (413)", resp.status_code == 413)
resp = send(c, "", [("x.gif", png(), "image/gif")])
check("unsupported declared type is rejected (400)", resp.status_code == 400)
resp = send(c, "", [("real.jpg", png(jpeg=True), "image/jpeg")])
check("real JPEG is accepted (200)", resp.status_code == 200)
resp = send(c, "", [("mismatch.png", png(jpeg=True), "image/png")])
check("JPEG content declared as PNG is rejected (400)", resp.status_code == 400)
resp = send(c, "", [("a.png", png(), "image/png")] * 5)
check("more than 4 photos is rejected (400)", resp.status_code == 400)

# ---------- feedback and reviews ----------
c = new_case()
send(c, "brown spots", [("ll.png", png(), "image/png")])
fb = client.post(f"/cases/{c}/feedback", json={"turn_index": 0, "helpful": True, "usefulness": 4})
check("feedback accepted for an existing turn", fb.status_code == 200)
check("feedback for a missing turn is rejected (400)",
      client.post(f"/cases/{c}/feedback", json={"turn_index": 9, "helpful": False}).status_code == 400)
check("feedback rejects usefulness outside 1-5 (422)",
      client.post(f"/cases/{c}/feedback", json={"helpful": True, "usefulness": 9}).status_code == 422)
rv = client.post(f"/cases/{c}/reviews", json={"turn_index": 0, "reviewer": "expert-1", "decision": "uncertain",
                                              "notes": "Needs a close-up."})
check("review accepted", rv.status_code == 200)
check("review rejects unknown decision (422)",
      client.post(f"/cases/{c}/reviews", json={"reviewer": "x", "decision": "maybe"}).status_code == 422)
stored = client.get(f"/cases/{c}").json()
check("case shows stored feedback", len(stored["feedback"]) == 1 and stored["feedback"][0]["helpful"] is True)
check("case shows stored review", len(stored["reviews"]) == 1 and stored["reviews"][0]["reviewer"] == "expert-1")
check("turn stores its turn_index", stored["turns"][0]["assessment"]["turn_index"] == 0)
check("unknown case returns 404 for feedback",
      client.post("/cases/nope/feedback", json={"helpful": True}).status_code == 404)

ev = client.get("/evidence/rec_ll")
check("GET /evidence/{id} returns the record", ev.status_code == 200 and ev.json()["type"] == "text")
q = client.get("/reviews/queue").json()
check("review queue lists cases", any(item["case_id"] == c for item in q["items"]))
cov = client.get("/knowledge/coverage").json()
check("coverage register includes Little Leaf",
      any(row["condition"] == "Little Leaf" for row in cov["register"]))
dash = client.get("/dashboard").json()
check("dashboard counts cases", dash["cases"] >= 1)
eval_r = client.post("/eval/run").json()
check("eval run reports source-id gate", eval_r["gates"]["source_ids_on_approved_records"] is True)
guide = client.get("/guidance/photos").json()
check("photo guidance explains why extra views matter",
      all("why" in v and "question" in v for v in guide["views"]))
listed = client.get("/cases").json()
check("GET /cases lists created cases", any(item["case_id"] == c for item in listed))
rr = client.post(f"/cases/{c}/request-review")
check("request review flags the case", rr.status_code == 200 and client.get(f"/cases/{c}").json()["review_requested"] is True)
gap_r = client.get("/knowledge/gaps").json()
check("unknown answers are logged as knowledge gaps", isinstance(gap_r["items"], list))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")