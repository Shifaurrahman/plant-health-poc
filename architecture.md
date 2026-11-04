# Plant Health Assistant — Architecture & End-to-End Guide

Version aligned with app `1.0.0` and plan v1.0 (25 October 2026).

This document explains **how the system works from start to finish**: which screen calls which API, which Python function calls the next, what data moves between layers, and what each file is responsible for.

> **Product boundary:** this is decision support, not a confirmed diagnosis. Visible symptoms can have infectious, pest, or environmental causes.

---

## 1. Big picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Browser (Vite + React)                                                 │
│  frontend/src/App.jsx                                                   │
│  Screens: New assessment · Case · Expert queue · Knowledge · Dashboard  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTP JSON / multipart
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FastAPI  —  backend/app/main.py                                        │
│  Validates requests → stores files → calls pipeline → returns JSON      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
         pipeline.py      store.py         routes_images.py
         (assess)         (SQLite +        (reference photos)
                          seed JSON)
                │
    ┌───────────┼───────────┬────────────┬────────────┐
    ▼           ▼           ▼            ▼            ▼
 routing    embedder   image_retrieval  retrieval    vlm
 photo_     evidence   validator        policy       versions
 guidance              eval_suite
```

**Startup command path**

1. `uvicorn app.main:app` loads `backend/app/main.py`.
2. Importing `from app.store import store` constructs a single `Store` instance:
   - opens/creates `backend/data/poc.sqlite3`
   - loads `backend/data/seed/records.json` into memory
   - loads `backend/data/seed/sources.json` into the `sources` table
3. Frontend `npm run dev` serves `frontend/src/main.jsx` → `App.jsx`, which talks to `http://127.0.0.1:8000`.

---

## 2. Repository map (what each file does)

### Frontend

| File | Role |
|---|---|
| `frontend/index.html` | HTML shell; title “Plant Health Assistant” |
| `frontend/src/main.jsx` | React entry: mounts `<App />` |
| `frontend/src/App.jsx` | Entire UI: screens, forms, API calls, result rendering |
| `frontend/src/index.css` | Layout and component styles |

### Backend core

| File | Role |
|---|---|
| `backend/app/main.py` | FastAPI routes; wires HTTP → services → store |
| `backend/app/config.py` | Paths, crop list, retrieval `K`, upload limits, versions |
| `backend/app/schemas.py` | Pydantic request bodies (`CaseCreate`, `FeedbackIn`, …) |
| `backend/app/store.py` | SQLite + in-memory records; persistence for cases/reviews/gaps |
| `backend/app/routes_images.py` | `GET /images/{path}` — safe serve of dataset reference photos |

### Assessment services (called from the pipeline)

| File | Role |
|---|---|
| `backend/app/services/pipeline.py` | **Orchestrator** — full assessment workflow |
| `backend/app/services/routing.py` | Chooses `image_assessment` / `text_qa` / `followup` |
| `backend/app/services/upload_validation.py` | File type, size, decode, quality; strips EXIF |
| `backend/app/services/photo_guidance.py` | Why to ask for whole-plant / close-up / underside |
| `backend/app/services/embedder.py` | Qwen3-VL-Embedding image + text embeddings (shared space; query/document split) |
| `backend/app/services/image_retrieval.py` | Image→image nearest neighbours in reference library |
| `backend/app/services/retrieval.py` | Keyword, semantic text, image→text search |
| `backend/app/services/evidence.py` | Merge channels; pick ~4–8 diverse evidence items |
| `backend/app/services/vlm.py` | Stub or Anthropic: `observe` then `synthesize` |
| `backend/app/services/validator.py` | Drop bad citations; cap questions; scrub chemicals |
| `backend/app/services/policy.py` | Chemical / dose / interval removal rules |
| `backend/app/services/versions.py` | Fingerprints (index SHA, records SHA, embedding model) |
| `backend/app/services/eval_suite.py` | Pilot release gates for `POST /eval/run` |

### Data & scripts

| Path | Role |
|---|---|
| `backend/data/seed/records.json` | Knowledge records (approved ones enter retrieval) |
| `backend/data/seed/sources.json` | Source metadata table |
| `backend/data/splits/main_index.csv` | Reference photo list (test set excluded) |
| `backend/data/embeddings/qwen3-vl-embedding-2b.npz` | Cached embedding vectors for reference photos |
| `backend/data/uploads/{case_id}/` | Farmer uploads (re-encoded JPEG, no EXIF) |
| `backend/data/poc.sqlite3` | Cases, turns, feedback, reviews, gaps |
| `backend/data/logs/knowledge_gaps.jsonl` | Append-only gap log |
| `backend/prompts/observe_v1.txt` | VLM observation prompt |
| `backend/prompts/synthesize_v4.txt` | VLM synthesis prompt |
| `backend/scripts/records_from_csv.py` | Expert CSV → validated `records.json` |
| `backend/scripts/make_splits.py` | Train/index vs held-out test splits |
| `backend/scripts/evaluate_retrieval.py` | Offline image retrieval accuracy |
| `backend/tests/smoke_test.py` | End-to-end API tests with fake embedder |

---

## 3. End-to-end farmer journey (happy path)

This is the main path the product is built around.

### Step A — Open the app

1. Browser loads Vite → `main.jsx` → `App`.
2. Default screen is **New assessment** (`screen === "assess"`).
3. `NewAssessment` mounts and calls:

```http
GET /guidance/photos
```

→ `main.photo_capture_guidance()` → `photo_guidance.capture_guidance()`  
→ returns three views with **why** each photo helps (whole-plant, close-up, underside).

### Step B — Create a case

User fills question, optional crop/region/duration/plant part/treatment, consent, and up to 4 photos. Submit runs:

```http
POST /cases
Content-Type: application/json
{ "crop": "brinjal", "region": "Sri Lanka", "plant_part": "leaf", "consent_reuse_photos": true, ... }
```

**Call chain**

```
App.NewAssessment.handleSubmit
  → main.create_case(CaseCreate)
    → store.create_case(meta)
      → INSERT into cases (case_id, meta, created_at, summary={}, review_requested=0)
      → returns { case_id, meta, created_at, turns: [] }
```

### Step C — Send first message (photos + text)

```http
POST /cases/{case_id}/messages
multipart/form-data: text=...  files=@leaf1.jpg  files=@leaf2.jpg
```

**Call chain (HTTP layer)**

```
App.NewAssessment.handleSubmit
  → main.add_message(case_id, text, files)
      1. _require_case → store.get_case
      2. for each file: upload_validation.validate_image
            sniff_type → decode → quality_issues → re-encode JPEG (strip EXIF)
      3. write files to data/uploads/{case_id}/{uuid}_{stem}.jpg
      4. pipeline.run_assessment(case, text, image_paths, quality)
      5. store.add_turn(...)
      6. store.set_summary(case_id, assessment.case_summary)
      7. return assessment JSON
  → onCreated(case_id) switches UI to Case screen
```

### Step D — Assessment pipeline (core AI workflow)

Everything below happens inside `pipeline.run_assessment`.

```
run_assessment(case, text, image_paths, quality_issues)
│
├─ get_vlm()                          # stub or Anthropic from PH_VLM_PROVIDER
├─ route(case, text, len(images))     # routing.py → image_assessment | text_qa | followup
│
├─ [if followup & no new images]
│     prior_images(case)              # reuse earlier upload paths
│
├─ OBSERVE (unless text_qa with no images)
│     vlm.observe(images, text, meta)
│       StubVLM  → placeholder strings
│       AnthropicVLM → prompts/observe_v1.txt + images → JSON observations
│
├─ IMAGE RETRIEVAL (if crop supported & not skip_vision)
│     for each usable photo:
│       image_retrieval.match_image(path)
│         embedder.embed_image_file(path)          # image vector
│         _reference_library()                     # load npz + main_index.csv
│         _approved_by_label()                     # only approved conditions
│         cosine similarity → top_k neighbours
│     candidates_from_matches(matches)
│     evidence_from_matches(matches)
│
├─ TEXT / MULTIMODAL RETRIEVAL
│     _query_text(text, meta, summary)
│     retrieval.search_channels(query, crop_filter, k=CHANNEL_K, image_paths=usable)
│         search_text          # keyword overlap
│         search_semantic      # text query↔record text
│         search_image_to_text # image↔record text
│
├─ decide_outcome(...)                # match | healthy | unknown | insufficient | ...
│
├─ select_evidence([image_ev, keyword, semantic, image_to_text])
│     evidence.py — dedupe, diversify conditions/types → ≤ EVIDENCE_K items
│
├─ _enrich(evidence)
│     store.get_source / get_record → titles, publishers, reference_text
│
├─ photo_guidance.quality_questions(...)
│
├─ SYNTHESIZE (unless pure text_qa)
│     vlm.synthesize(meta, text, observations, evidence, outcome, summary)
│       None → fallback_draft(...)                 # template answer from retrieval
│       else → apply_fit_check (poor fit → conflict)
│            → _maybe_second_round (≤1 extra retrieval)
│
├─ Build limitations list (base + quality + VLM errors + outcome notes)
├─ validator.validate(draft, evidence)
│     drop unknown evidence_ids
│     policy.scrub on all text fields (chemicals/doses/intervals)
│
├─ current_versions(...)              # app, policy, embedding, index SHA, records SHA
├─ [if no coverage] store.log_gap(...)
│
└─ return structured assessment JSON
```

### Step E — UI renders the answer

`CaseScreen` loads:

```http
GET /cases/{case_id}
```

→ shows turns with timestamps; latest turn uses `AssessmentBody`:

| UI block | Assessment field |
|---|---|
| Outcome banner | `outcome`, `route` |
| What the photo shows | `observations` |
| Coverage + next step | `coverage`, `next_step` |
| How to read this result | `signals` (three separate signals) |
| Candidate causes | `candidate_causes` (+ `general_knowledge` marked) |
| Alternatives | `alternatives` |
| Reference photos | `retrieved_evidence` where `type=image` → `GET /images/...` |
| Cited records | `retrieved_evidence` where `type=text` |
| Please add | `questions` (max 3) |
| Limitations | `limitations` |
| Technical details | `versions`, `latency_ms`, `policy_removals` |

### Step F — Follow-up in the same case

User sends more text and/or photos from `CaseScreen.sendFollowUp` → same `POST /cases/{id}/messages`.

Routing rules (`routing.route`):

| Situation | Route |
|---|---|
| New images attached | `image_assessment` |
| Case has history + management wording (prevent/treat/…) | `text_qa` |
| Case has history, otherwise | `followup` (reuses prior images if none new) |
| First turn, text only | `text_qa` |
| First turn, images only | `image_assessment` |

For `text_qa` without new images, the pipeline **skips VLM observe/synthesize inventing an image assessment** and retrieves text knowledge directly.

### Step G — Feedback / expert review

```
FeedbackForm → POST /cases/{id}/feedback → store.add_feedback
ReviewForm   → POST /cases/{id}/reviews  → store.add_review
Request btn  → POST /cases/{id}/request-review → store.flag_review
```

Expert queue screen:

```
QueueScreen → GET /reviews/queue → _queue_item for each case, sorted by priority
```

Priority scores disputes, review requests, missing knowledge, image quality, unreviewed status.

---

## 4. Assessment response contract

Every `POST .../messages` returns roughly:

```json
{
  "route": "image_assessment",
  "outcome": "match",
  "observations": ["..."],
  "candidate_causes": [
    {
      "name": "Little Leaf",
      "fit": "partial",
      "support": ["..."],
      "against_or_missing": ["..."],
      "general_knowledge": ["..."],
      "evidence_ids": ["Raw_Dataset/..."]
    }
  ],
  "alternatives": ["..."],
  "coverage": "partial",
  "signals": {
    "retrieval_relevance": { "best_reference_similarity": 0.87, "meaning": "..." },
    "evidence_coverage": "partial",
    "diagnostic_confidence": { "value": null, "meaning": "Not assessed..." }
  },
  "limitations": ["..."],
  "questions": ["..."],
  "next_step": "...",
  "retrieved_evidence": [ { "id": "...", "type": "image|text", "channel": "...", ... } ],
  "policy_removals": 0,
  "diagnostics": { "route": "...", "text_channels": { ... } },
  "latency_ms": { "total": 1234, "image_match": 800, "text_match": 50 },
  "versions": { "app": "1.0.0", "embedding": "Qwen/Qwen3-VL-Embedding-2B", "index_sha": "...", ... },
  "case_summary": { "candidates": ["..."], "last_outcome": "...", ... },
  "turn_index": 0
}
```

**Three signals are never mixed**

1. **Retrieval relevance** — how close a reference matched (visual similarity).
2. **Evidence coverage** — whether reviewed knowledge supports the answer (`none` / `partial`; never auto-`strong`).
3. **Diagnostic confidence** — always `null` until calibrated on expert-labelled cases.

**Outcomes** (`pipeline.decide_outcome` + fit check)

| Outcome | Meaning |
|---|---|
| `match` | Clear majority of neighbours agree on one non-healthy class |
| `healthy_appearance` | Majority class is Healthy |
| `unknown_cause` | Neighbours split across classes |
| `insufficient_image` | All uploads failed quality checks |
| `unsupported_crop` | Crop not in `SUPPORTED_CROPS` |
| `text_only_match` | No/skipped images; text retrieval hit |
| `no_match` | Nothing in approved knowledge |
| `conflict` | VLM marked all candidates `fit: poor` |

---

## 5. Retrieval channels in detail

Configured in `config.py`: `CHANNEL_K=15`, `EVIDENCE_K=8`, `TOP_K=10`.

| Channel | Module / function | Input | Output |
|---|---|---|---|
| Image→image | `image_retrieval.match_image` | Upload path | Top-k reference photos with similarity |
| Text keyword | `retrieval.search_text` | Query string | Records with overlapping non-stopword tokens |
| Text semantic | `retrieval.search_semantic` | Query string | Records by text cosine |
| Image→text | `retrieval.search_image_to_text` | Upload path(s) | Records by image↔text cosine |

`retrieval.search_channels` runs keyword always; semantic and image→text are try/except so embedding failures do not break the HTTP response.

`evidence.select_evidence` then:

1. Dedupes by evidence `id`
2. Marks whether a source is an *independent* source (same source not counted twice)
3. Prefers diverse conditions and a mix of image + text
4. Caps at `EVIDENCE_K`

Only records with `review_status == "approved"` and not `retired` enter trusted retrieval (`store.approved_records`).

---

## 6. Data model (how entities link)

```
Crop (pilot: brinjal)
  └── Condition (e.g. Little Leaf)  ← condition_id / category
        └── Knowledge record (record_id)  ← symptoms, guidance, source_id, review_status
              ├── Reference images (image_path, label_status)
              └── Source (source_id)  ← title, URL, publisher, date, licence

Case (case_id)
  ├── meta (crop, region, plant_part, consent, ...)
  ├── summary (rolling case summary for follow-ups)
  ├── turns[]  → text, images[], assessment{}, at
  ├── feedback[]
  └── reviews[]  → decision, confirmation_method, publish_decision
```

**Label distinctions** (stored on records / evidence):  
`expert_confirmed`, `lab-confirmed` (via review confirmation_method), `expert_suspected`, user-reported (feedback), model-suggested (`general_knowledge`).

Missing fields stay null/empty. The model must not invent them as facts; VLM grounding uses only `observations` + `reference_text`.

---

## 7. File-by-file call reference

### `main.py` endpoints → functions

| Endpoint | Handler | Calls |
|---|---|---|
| `GET /health` | `health` | `store.approved_count` |
| `GET /guidance/photos` | `photo_capture_guidance` | `capture_guidance` |
| `POST /cases` | `create_case` | `store.create_case` |
| `GET /cases` | `list_cases` | `store.list_case_ids`, `_queue_item` |
| `GET /cases/{id}` | `get_case` | `_require_case` → `store.get_case` |
| `PATCH /cases/{id}` | `patch_case` | `store.update_case_meta` |
| `POST /cases/{id}/messages` | `add_message` | `validate_image` → `run_assessment` → `add_turn` / `set_summary` |
| `POST /cases/{id}/feedback` | `add_feedback` | `store.add_feedback` |
| `POST /cases/{id}/reviews` | `add_review` | `store.add_review` |
| `POST /cases/{id}/request-review` | `request_review` | `store.flag_review` |
| `GET /reviews/queue` | `review_queue` | `_queue_item` sorted by priority |
| `GET /knowledge/records` | `list_records` | `store.all_records`, `list_sources` |
| `PATCH /knowledge/records/{id}` | `patch_record` | `store.update_record` (writes `records.json`) |
| `GET /knowledge/coverage` | `coverage` | `store.coverage_register` |
| `GET /knowledge/gaps` | `gaps` | `store.list_gaps` |
| `GET /evidence/{id}` | `get_evidence` | `store.get_record` / scan image paths |
| `GET /dashboard` | `dashboard` | `store.dashboard` |
| `POST /eval/run` | `eval_run` | `eval_suite.run_eval` |
| `GET /images/{path}` | `routes_images.get_image` | FileResponse under dataset root |
| `GET /uploads/{case}/{name}` | `get_upload` | FileResponse under uploads |

### `pipeline.py` helpers

| Function | Purpose |
|---|---|
| `run_assessment` | Full orchestrator |
| `route` / `prior_images` | (imported) request type + reuse photos |
| `decide_outcome` | Map retrieval state → outcome label |
| `apply_fit_check` | Downgrade / conflict from VLM `fit` |
| `fallback_draft` | Template answer when VLM returns `None` |
| `_enrich` | Attach source + reference_text |
| `_maybe_second_round` | Bounded second retrieval |
| `_case_summary` | Compact state for next turn |
| `record_limitations` | Pull limitations from matched records |

### `vlm.py`

| Function / class | Purpose |
|---|---|
| `get_vlm()` | Factory: `StubVLM` or `AnthropicVLM` |
| `StubVLM.observe` | Placeholder observations |
| `StubVLM.synthesize` | Always `None` → pipeline fallback |
| `AnthropicVLM.observe` | Vision call with `observe_v1.txt` |
| `AnthropicVLM.synthesize` | Text call with `synthesize_v4.txt` |
| `_normalise_draft` | Keep only allowed condition names & evidence ids; split ungrounded / hedged claims into `general_knowledge` |

Farmer photos are **not** sent to Anthropic unless `PH_VLM_ALLOW_REAL_PHOTOS=yes`.

### `validator.py` + `policy.py`

```
validator.validate(draft, evidence)
  → for each cause: filter evidence_ids ∈ retrieved set
  → policy.scrub on support, against, alternatives, questions, limitations, next_step
       chemical names / dose patterns / spray intervals removed
       unless the same phrase appears in approved record text
```

### Frontend components → APIs

| Component | APIs used |
|---|---|
| `NewAssessment` | `GET /guidance/photos`, `POST /cases`, `POST /cases/{id}/messages` |
| `CaseScreen` | `GET /cases/{id}`, `POST .../messages`, `POST .../request-review`, `GET /uploads/...` |
| `AssessmentBody` | (render only) + `GET /images/...` for refs |
| `FeedbackForm` | `POST .../feedback` |
| `ReviewForm` | `POST .../reviews` |
| `QueueScreen` | `GET /reviews/queue` |
| `KnowledgeScreen` | `GET /knowledge/records`, `GET /knowledge/coverage`, `PATCH /knowledge/records/{id}` |
| `DashboardScreen` | `GET /dashboard`, `GET /knowledge/gaps`, `POST /eval/run` |

---

## 8. Other user journeys (same architecture)

### Text-only advice

1. New case (or follow-up with management wording) + question, no photos.
2. Route = `text_qa`.
3. Pipeline skips observe; runs `search_channels` without requiring image matches.
4. Outcome `text_only_match` or `no_match`; if none, `log_gap`.

### Expert knowledge loop

1. Case appears in **Expert queue** (dispute / gap / request).
2. Reviewer opens case → `ReviewForm` saves decision + `publish_decision`.
3. **Knowledge manager** approves/retires records (`PATCH`); only approved enter retrieval.
4. Coverage register shows text/image/management gaps.
5. `POST /eval/run` checks release gates (source ids, no TODO in approved records).

### Offline data prep (scripts, not request path)

```
raw photos + metadata.csv
  → scripts/make_splits.py        → main_index.csv + main_test.csv
  → scripts/evaluate_retrieval.py → builds/uses the model's embedding cache; reports top-1 accuracy
expert CSV
  → scripts/records_from_csv.py   → records.json (validated)
```

Test photos stay out of `main_index.csv` so retrieval evaluation is fair.

---

## 9. Sequence diagram (one assessment)

```
Farmer          App.jsx           main.py           pipeline.py         retrieval*        store
  │                │                 │                   │                   │               │
  │ submit form    │                 │                   │                   │               │
  │───────────────>│ POST /cases     │                   │                   │               │
  │                │────────────────>│ create_case ─────────────────────────────────────────>│
  │                │<────────────────│ case_id           │                   │               │
  │                │ POST /messages  │                   │                   │               │
  │                │────────────────>│ validate_image    │                   │               │
  │                │                 │ run_assessment ──>│                   │               │
  │                │                 │                   │ route()           │               │
  │                │                 │                   │ observe()         │               │
  │                │                 │                   │ match_image ─────>│ embedder      │
  │                │                 │                   │ search_channels ─>│ keyword/sem/i2t│
  │                │                 │                   │ select_evidence   │               │
  │                │                 │                   │ synthesize/fallback               │
  │                │                 │                   │ validate+scrub    │               │
  │                │                 │                   │ log_gap? ────────────────────────>│
  │                │                 │<──────────────────│ assessment JSON   │               │
  │                │                 │ add_turn / set_summary ─────────────────────────────>│
  │                │<────────────────│ assessment        │                   │               │
  │ render answer  │                 │                   │                   │               │
  │<───────────────│                 │                   │                   │               │
```

---

## 10. Configuration & environment

| Variable / constant | Where | Effect |
|---|---|---|
| `PH_DATA_DIR` | config | Root for seed, uploads, sqlite, embeddings |
| `PH_VLM_PROVIDER` | config / vlm | `stub` (default) or `anthropic` |
| `ANTHROPIC_API_KEY` | env | Required for Anthropic |
| `PH_VLM_MODEL` | env | Anthropic model id |
| `PH_VLM_ALLOW_REAL_PHOTOS` | env | Must be `yes` to send farmer uploads to API |
| `HF_HOME` | env | Hugging Face cache location for the embedding model |
| `PH_EMBED_MODEL` | env | Embedding model id or local path (default `Qwen/Qwen3-VL-Embedding-2B`) |
| `PH_QWEN_REPO_DIR` | env | Folder of the cloned QwenLM/Qwen3-VL-Embedding repo (provides `Qwen3VLEmbedder`) |
| `SUPPORTED_CROPS` | config | Pilot crop list (currently `brinjal`) |
| `CHANNEL_K` / `EVIDENCE_K` / `TOP_K` | config | Retrieval sizes (tuning start points, not diagnostic thresholds) |
| `MAX_RETRIEVAL_ROUNDS` | config | Extra synthesis retrieval passes (MVP = 1) |

---

## 11. Safety, privacy, and failure behaviour

- Uploads: magic-byte type check, 8 MB limit, max 4 images, quality flags, EXIF stripped.
- Path traversal blocked on `/images` and `/uploads`.
- Chemical product names, doses, harvest intervals scrubbed unless present in approved records.
- VLM failures → template `fallback_draft`; image match failures → limitation strings, request continues.
- Semantic / image-to-text embedding errors → empty channel + diagnostic note, not HTTP 500.
- Cases private by default (local SQLite); consent checkbox for photo storage; reuse as reference needs expert publish decision.
- No authentication yet — local POC only; do not expose to the internet.

---

## 12. How to read the code in order

If you are new to this repo, open files in this sequence:

1. `frontend/src/App.jsx` — user-visible flow  
2. `backend/app/main.py` — HTTP surface  
3. `backend/app/services/pipeline.py` — assessment brain  
4. `backend/app/services/routing.py` → `image_retrieval.py` → `retrieval.py` → `evidence.py`  
5. `backend/app/services/vlm.py` + `prompts/`  
6. `backend/app/services/validator.py` + `policy.py`  
7. `backend/app/store.py` — persistence  
8. `backend/tests/smoke_test.py` — executable specification of expected behaviour  

For a live walkthrough: start backend + frontend, create a case with a held-out test photo from `main_test.csv`, then inspect the returned `diagnostics`, `versions`, and `retrieved_evidence` fields.