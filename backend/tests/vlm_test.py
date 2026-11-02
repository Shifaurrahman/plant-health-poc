#!/usr/bin/env python3
"""Tests for the Anthropic VLM adapter, using a fake client (no network, no key needed).

Run from backend/:  python tests/vlm_test.py
"""
import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ["PH_DATA_DIR"] = str(TMP / "data")
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("PH_VLM_ALLOW_REAL_PHOTOS", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402
import app.services.vlm as vlm_mod  # noqa: E402

failures = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        failures.append(name)


class FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)

        class R:
            content = [FakeBlock(out)]
        return R()


def make_vlm(outputs):
    os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
    os.environ["PH_VLM_MODEL"] = "test-model"
    v = vlm_mod.AnthropicVLM()
    v.client = FakeClient(outputs)
    return v


EVIDENCE = [
    {"id": "Raw_Dataset/Little_Leaf/a.jpg", "type": "image", "title": "Little Leaf reference photo",
     "condition": "Little Leaf", "source_title": "Test source", "review_status": "approved"},
    {"id": "dev_healthy", "type": "text", "title": "Healthy", "source": "src", "status": "approved"},
]

# 1) valid JSON: allowed names and cited ids survive, invented ones are dropped
v = make_vlm([json.dumps({
    "candidate_causes": [
        {"name": "Little Leaf", "support": ["small leaves"], "against_or_missing": [],
         "evidence_ids": ["Raw_Dataset/Little_Leaf/a.jpg", "made_up_id"]},
        {"name": "Tomato Blight", "support": [], "against_or_missing": [], "evidence_ids": ["dev_healthy"]},
    ],
    "alternatives": ["Healthy", "Invented"], "questions": ["Q1", "Q2", "Q3", "Q4"], "next_step": "Check it.",
})])
d = v.synthesize({"crop": "brinjal"}, "q", ["obs"], EVIDENCE, "match")
if d is None:
    sys.exit(f"synthesis returned None. Reason: {v.last_error}. Check that prompts/synthesize_v2.txt exists next to app/.")
check("invented condition name is dropped", [c["name"] for c in d["candidate_causes"]] == ["Little Leaf"])
check("invented evidence id is dropped", d["candidate_causes"][0]["evidence_ids"] == ["Raw_Dataset/Little_Leaf/a.jpg"])
check("alternatives keep only evidence names", d["alternatives"] == ["Healthy"])
check("questions capped at 3", len(d["questions"]) == 3)

# 2) fenced JSON is accepted
v = make_vlm(["```json\n" + json.dumps({"candidate_causes": [], "alternatives": [], "questions": [],
                                         "next_step": "x"}) + "\n```"])
d = v.synthesize({}, "", [], EVIDENCE, "match")
check("fenced JSON is parsed", d is not None and d["next_step"] == "x")

# 3) invalid JSON -> None and a visible reason
v = make_vlm(["sorry, here is some prose instead of JSON"])
d = v.synthesize({}, "", [], EVIDENCE, "match")
check("invalid JSON falls back (None)", d is None)
check("fallback reason is recorded", v.last_error and "template answer used" in v.last_error)

# 4) unknown_cause never returns candidate names
v = make_vlm([json.dumps({"candidate_causes": [{"name": "Little Leaf", "evidence_ids": ["dev_healthy"]}],
                          "next_step": "n"})])
d = v.synthesize({}, "", [], EVIDENCE, "unknown_cause")
check("unknown_cause forces an empty candidate list", d["candidate_causes"] == [])

# 5) no evidence -> no API call at all
v = make_vlm([])
d = v.synthesize({}, "", [], [], "no_match")
check("no evidence means no API call", d is None and v.client.calls == [])

# 6) farmer photos are blocked unless explicitly allowed
# .env may set this to yes for dataset testing; the consent test must not depend on it
os.environ["PH_VLM_ALLOW_REAL_PHOTOS"] = "no"
upload_dir = Path(os.environ["PH_DATA_DIR"]) / "uploads" / "case1"
upload_dir.mkdir(parents=True, exist_ok=True)
farmer = upload_dir / "farmer.jpg"
Image.new("RGB", (300, 300), (30, 120, 40)).save(farmer, format="JPEG")
v = make_vlm([])
out = v.observe([str(farmer)], "q", {})
check("farmer photo is not sent without consent", out == [] and v.client.calls == [])
check("consent reason is recorded", v.last_error and "consent" in v.last_error)

# 7) dataset photo outside uploads is sent, resized, and re-encoded as JPEG
dataset = TMP / "dataset.png"
Image.new("RGB", (2000, 1500), (10, 200, 10)).save(dataset, format="PNG")
v = make_vlm([json.dumps({"observations": ["green leaf"]})])
out = v.observe([str(dataset)], "q", {"crop": "brinjal"})
block = v.client.calls[0]["messages"][0]["content"][0]
check("observation parsed", out == ["green leaf"])
check("image sent as JPEG base64", block["source"]["media_type"] == "image/jpeg")
raw = base64.b64decode(block["source"]["data"])
sent = Image.open(io.BytesIO(raw))
check("image resized to at most 1024 px", max(sent.size) <= 1024)

# 8) no key -> stub with a configuration message, never a crash
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ["PH_VLM_PROVIDER"] = "anthropic"
g = vlm_mod.get_vlm()
check("missing key falls back to stub", g.name == "stub")
check("missing key is explained", g.config_error and "ANTHROPIC_API_KEY" in g.config_error)

# 9) prompt files exist
check("observe prompt file exists", (vlm_mod.PROMPTS_DIR / "observe_v1.txt").exists())
check("synthesize prompt file exists", (vlm_mod.PROMPTS_DIR / "synthesize_v4.txt").exists())

# 15) grounding: a sentence with no word from observations or reference text is moved out
EVID_REF = [dict(EVIDENCE[1], reference_text="Leaves become small and narrow")]
v = make_vlm([json.dumps({"candidate_causes": [{"name": "Healthy",
    "support": ["Leaves look small and narrow", "Reference photos show purple veins"],
    "against_or_missing": [], "evidence_ids": ["dev_healthy"]}], "next_step": "n"})])
c = v.synthesize({}, "", ["Lower leaves show brown spots"], EVID_REF, "match")["candidate_causes"][0]
check("sentence from reference text stays as support", "Leaves look small and narrow" in c["support"])
check("invented detail about reference photos is moved out",
      "Reference photos show purple veins" not in c["support"]
      and any("purple" in g for g in c["general_knowledge"]))

# 10) general knowledge is kept apart and scrubbed for chemical names
v = make_vlm([json.dumps({"candidate_causes": [{"name": "Little Leaf", "support": ["small leaves"],
    "against_or_missing": [], "general_knowledge": ["Use mancozeb 2 ml per litre"],
    "evidence_ids": ["dev_healthy"]}], "next_step": "n"})])
d = v.synthesize({}, "", [], EVIDENCE, "match")
gk = d["candidate_causes"][0]["general_knowledge"]
check("general knowledge is returned separately", len(gk) == 1)
check("chemical name and dose removed from general knowledge", "mancozeb" not in gk[0].lower() and "2 ml" not in gk[0].lower())

# 11) fit values are kept; unknown fit becomes partial
v = make_vlm([json.dumps({"candidate_causes": [
    {"name": "Little Leaf", "fit": "poor", "evidence_ids": ["dev_healthy"]},
    {"name": "Healthy", "fit": "maybe", "evidence_ids": ["dev_healthy"]}], "next_step": "n"})])
d = v.synthesize({}, "", [], EVIDENCE, "match")
fits = {c["name"]: c["fit"] for c in d["candidate_causes"]}
check("fit poor is kept", fits.get("Little Leaf") == "poor")
check("unknown fit becomes partial", fits.get("Healthy") == "partial")

# 12) hedged statements move from support to general knowledge
v = make_vlm([json.dumps({"candidate_causes": [{"name": "Little Leaf",
    "support": ["Typically causes leaf size reduction", "Lower leaves show brown spots"],
    "against_or_missing": [], "evidence_ids": ["dev_healthy"]}], "next_step": "n"})])
c = v.synthesize({}, "", [], EVIDENCE, "match")["candidate_causes"][0]
check("hedged sentence is not an observation", c["support"] == ["Lower leaves show brown spots"])
check("hedged sentence is kept as general knowledge",
      any("leaf size reduction" in g for g in c["general_knowledge"]))

# 13) farmer questions drop lab and microscope wording
v = make_vlm([json.dumps({"candidate_causes": [], "questions": [
    "Are the spots wet or dry?", "Are spores visible under a microscope?", "Did it start after rain?"],
    "next_step": "n"})])
d = v.synthesize({}, "", [], EVIDENCE, "match")
check("lab and microscope questions are dropped",
      d["questions"] == ["Are the spots wet or dry?", "Did it start after rain?"])

# 14) pipeline: all candidates poor -> conflict outcome, no candidate suggested
import app.services.pipeline as pl  # noqa: E402


class FakeVLM:
    name = "anthropic"
    model_name = "fake"
    prompt_version = "test"
    config_error = None

    def errors(self):
        return []

    def observe(self, *a, **k):
        return ["lower leaves brown"]

    def synthesize(self, *a, **k):
        return {"candidate_causes": [{"name": "Little Leaf", "fit": "poor", "support": [],
                                      "against_or_missing": ["spots do not fit"], "general_knowledge": [],
                                      "evidence_ids": ["dev_x"]}],
                "alternatives": [], "questions": [], "next_step": "x"}


pl.get_vlm = lambda: FakeVLM()
pl.retrieval.search_text = lambda q, crop=None, k=8: [
    {"id": "dev_x", "type": "text", "title": "Little Leaf", "source": "src", "status": "approved",
     "matched_terms": ["spots"]}]
r = pl.run_assessment({"meta": {"crop": "brinjal"}}, "brown spots", [], [])
check("all-poor candidates give outcome conflict", r["outcome"] == "conflict")
check("conflict suggests no candidate", r["candidate_causes"] == [])
check("conflict names the rejected class in limitations",
      any("'Little Leaf'" in l for l in r["limitations"]))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL VLM TESTS PASSED")