#!/usr/bin/env python3
"""Prints the expert verdict for a healthy-looking photo. Diagnostic only, not a test."""
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

TMP = Path(tempfile.mkdtemp())
os.environ["PH_DATA_DIR"] = str(TMP / "data")
os.environ["PH_VLM_PROVIDER"] = "stub"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.services.image_retrieval as ir  # noqa: E402
from app.services import expert  # noqa: E402,F401
from app.services.vlm import StubVLM  # noqa: E402
import app.services.pipeline as pipeline  # noqa: E402
import app.services.retrieval as tr  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402

pipeline.get_vlm = lambda: StubVLM()

# Healthy-looking query vector: near the Healthy reference vectors.
def fake_embed(path):
    return np.array([0.0, 0.0, 1.0])

ir.embed_image_file = fake_embed
tr.embed_image_file = fake_embed
tr.encode_document = lambda ts: np.stack([np.array([1.0, 0.0, 0.0]) for _ in ts])
tr.encode_query = lambda ts: np.stack([np.array([1.0, 0.0, 0.0]) for _ in ts])

client = TestClient(fastapi_app)
case_id = client.post("/cases", json={"crop": "brinjal", "region": "Sri Lanka"}).json()["case_id"]

buf = io.BytesIO()
Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)).save(buf, "PNG")
resp = client.post(f"/cases/{case_id}/messages", data={"text": ""},
                   files={"files": ("healthy.png", buf.getvalue(), "image/png")})
print("status:", resp.status_code)
body = resp.json()
expert_block = body.get("expert", {}) if isinstance(body, dict) else {}
print("outcome:          ", body.get("outcome") if isinstance(body, dict) else body)
print("expert verdict:   ", expert_block.get("verdict"))
print("expert candidates:", expert_block.get("candidates"))
print("photo passing:    ", (body.get("diagnostics") or {}).get("photo_matches_passing") if isinstance(body, dict) else None)
