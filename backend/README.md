# Plant Health Assistant — backend

Python FastAPI service implementing the plan v1.0 workflow: deterministic routing,
multimodal retrieval, structured VLM outputs (stub or Anthropic), expert review and
knowledge publication.

## Run

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Docs: http://127.0.0.1:8000/docs  
Health: http://127.0.0.1:8000/health

## Smoke tests

```bash
python tests/smoke_test.py
```

Uses a fake embedder (no CLIP download).

## Knowledge

1. Copy `data/seed/records.example.json` → `data/seed/records.json` (or convert CSV with `scripts/records_from_csv.py`).
2. Copy `data/seed/sources.example.json` → `data/seed/sources.json`.
3. Set `review_status` to `approved` only after expert sign-off.
4. Restart the server after replacing the seed file. In-app approve/retire writes back to `records.json`.

## Evaluation fixtures

Copy `data/eval/cases.example.json` → `data/eval/cases.json` for `POST /eval/run`.
Keep evaluation images out of `main_index.csv`.

## Key modules

| Module | Role |
|---|---|
| `services/routing.py` | image_assessment / text_qa / followup |
| `services/pipeline.py` | observe → retrieve → synthesise → validate |
| `services/evidence.py` | merge channels, diversify 4–8 evidence items |
| `services/photo_guidance.py` | why whole-plant / close-up / underside help |
| `services/eval_suite.py` | release-gate checks |
| `store.py` | SQLite cases, reviews, gaps + seed records |


