# Plant Health Assistant (POC)

Conversational decision-support for plant health questions from images and/or text.
Implements the Version 1.0 plan: multimodal retrieval, structured assessments, expert review,
knowledge administration and an evaluation suite. Pilot crop assumption: **brinjal** (confirm in discovery).

> **Decision support only.** Visible symptoms can have infectious, pest or environmental causes.
> The system does not confirm a diagnosis. Some cases need field inspection or lab testing.

**Full system walkthrough (files, functions, call graph):** see [`architecture.md`](architecture.md).

---

## Status (aligned to plan v1.0)

| Area | Status |
|---|---|
| Case workflow (create, multi-photo, follow-up, summary) | Working |
| Routing (image assessment / text Q&A / follow-up) | Working |
| Image-to-image retrieval (CLIP) | Working |
| Text keyword + semantic + image-to-text channels | Working (semantic needs CLIP; fails soft if unavailable) |
| Evidence merge / diversity selection | Working |
| Structured answer + three signals (relevance / coverage / confidence) | Working |
| Unknown outcomes (healthy, unknown, insufficient, unsupported crop, conflict) | Working |
| Photo quality checks + capture guidance | Working |
| Policy scrub (chemicals / doses / intervals) | Working |
| Expert review queue + feedback + knowledge gaps | Working |
| Knowledge manager / coverage register / publish-retire | Working |
| Quality dashboard + `POST /eval/run` + `GET /evidence/{id}` | Working |
| Frontend screens (assessment, case, queue, knowledge, dashboard) | Working |
| Expert-approved seed records | **Pending** — use reviewed CSV; test records only for DEV |
| VLM observe + synthesise | Stub by default; Anthropic adapter when configured |
| Auth / PostgreSQL / object storage | Deferred (local SQLite POC) |

**Important:** Until an expert approves records, the system returns "no approved match" for every
question. That is correct behaviour.

---

## Repository layout

```
plant-health-poc/
├── README.md
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── README.md                    (backend notes)
│   ├── app/
│   │   ├── main.py                  FastAPI app and endpoints
│   │   ├── config.py                all paths and constants
│   │   ├── schemas.py               request models
│   │   ├── store.py                 cases and turns in SQLite; records loaded at startup
│   │   ├── routes_images.py         serves dataset photos at /images/...
│   │   └── services/
│   │       ├── embedder.py          CLIP image embedder (loaded on first use)
│   │       ├── image_retrieval.py   photo -> top-k reference photos
│   │       ├── retrieval.py         keyword matching over approved records
│   │       ├── pipeline.py          builds the answer
│   │       ├── validator.py         checks cited IDs, caps questions, scrubs forbidden content
│   │       ├── policy.py            chemical / dose / interval rules
│   │       └── vlm.py               VLM adapter (stub for now)
│   ├── scripts/
│   │   ├── download_datasets.sh     downloads datasets (run on your own machine)
│   │   ├── build_manifest.py        lists every image in a class-folder dataset
│   │   ├── make_splits.py           main split and leave-one-farm-out splits
│   │   ├── evaluate_retrieval.py    offline retrieval accuracy (top-1 / top-5)
│   │   └── records_from_csv.py      expert CSV -> data/seed/records.json (validated)
│   └── data/                        (mostly git-ignored)
│       ├── dataset_sources.md       where each dataset comes from, and its caveats
│       ├── seed/
│       │   ├── records.example.json     schema example
│       │   ├── expert_records_template.csv  blank template for the expert
│       │   ├── sample_test_records.csv     DEV_TEST records (not real advice)
│       │   └── records.json             generated; git-ignored
│       ├── raw/                     dataset photos; git-ignored
│       ├── splits/                  generated split CSVs; git-ignored
│       ├── embeddings/              cached CLIP vectors; git-ignored
│       ├── results/                 evaluation outputs; git-ignored
│       ├── uploads/                 user photos; git-ignored
│       └── poc.sqlite3              case storage; git-ignored
└── frontend/
    ├── package.json
    ├── index.html
    └── src/
        ├── App.jsx                  the whole UI
        └── index.css
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 | Tested on 3.11 (Windows) |
| Node.js + npm | 18 or newer | Needed for the frontend |
| Git | any | For version control |
| Disk space | about 5 GB free on the drive holding the project and caches | Dataset is about 2.4 GB zip; CLIP model is about 605 MB |
| Internet | needed once | To download the CLIP model and datasets |

**Windows note:** If drive C: is low on space, keep the Hugging Face cache and pip temp files on
another drive. See [Troubleshooting](#troubleshooting).

---

## 1. Dataset setup (not included in the repo)

The dataset is **not in git**. It is too large, and its licence must be checked on each source page.

Sources and caveats are listed in `backend/data/dataset_sources.md`.

**Expected folder layout:**

```
backend/data/raw/sri_lanka_brinjal/
├── metadata.csv
├── Raw_Dataset/          1,459 original photos, 8 class folders (used by the app)
├── Phase_I_Dataset/      originals + random copies (only for robustness checks)
└── Phase_II_Dataset/     originals + augmented copies (only for robustness checks)
```

**Download:**
- Brinjal: SLIF-Brinjal from Mendeley (version 2). Download `SLIF-Brinjal.zip` (2.35 GB)
  and `metadata.csv`, then extract the zip into `backend\data\raw\sri_lanka_brinjal\`.
  After extraction, make sure `Raw_Dataset`, `Phase_I_Dataset`, and `Phase_II_Dataset` sit
  directly inside `sri_lanka_brinjal\`, not nested one level deeper.
- Tea and Kaggle datasets are listed in `dataset_sources.md`. They are not used by the app yet.

**Generated splits** (run once, after `metadata.csv` is in place):

```cmd
cd backend
python scripts\make_splits.py data\raw\sri_lanka_brinjal\metadata.csv data\splits
```

This writes `main_test.csv` and `main_index.csv` (used by the app), plus the leave-one-farm-out files.

---

## 2. Backend setup

Run all backend commands from the `backend` folder.

```cmd
cd "E:\Desktop-Copy\Arva Tech\plant-health-poc\backend"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch open_clip_torch pillow numpy
```

`requirements.txt` covers the web server only. The last line adds the image-matching packages.
`torch` is large (about 2 GB). If the install fails with "No space left on device", set the temp
folder to a drive with space first:

```cmd
mkdir E:\pip_tmp
set TMP=E:\pip_tmp
set TEMP=E:\pip_tmp
set PIP_CACHE_DIR=E:\pip_cache
```

### Start the backend

```cmd
set HF_HOME=E:\hf_cache
uvicorn app.main:app --reload --port 8000
```

- `HF_HOME` moves the CLIP model cache off drive C:. Set it in every window where you start uvicorn.
- The first image upload downloads the CLIP model (about 605 MB) if it is not cached.
  This can take several minutes.
- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Expected health output:

```json
{"status": "ok", "vlm_provider": "stub", "approved_records": 8}
```

`approved_records` is `0` until expert records are approved.

---

## 3. Expert knowledge records

Records are the knowledge base. Only records with `review_status = approved` are used.

### Workflow

1. The expert fills in `backend/data/seed/expert_records_template.csv` (one row per class).
   Save as UTF-8 CSV. Use `;` to separate list items such as `alternatives` and `limitations`.
2. Convert it, with validation:
   ```cmd
   python scripts\records_from_csv.py data\seed\expert_records.csv data\seed\records.json
   ```
   If there are errors, nothing is written, and each error shows its line number.
3. Restart uvicorn, and check `/health` for the approved count.

### Rules enforced by the converter

- `condition` must be one of the 8 classes: Bacterial Blight, Bacterial Leaf Spot, Bacterial Wilt,
  Cercospora Leaf Spot, Healthy, Little Leaf, Mosaic Virus, Powdery Mildew.
- `category`: disease, pest, abiotic, or healthy.
- `review_status = approved` needs `reviewed_by`, `reviewed_on`, and no `TODO` text.
- Chemical names in `management_general` produce a warning. The expert must confirm they come from the source.

### Fields that change the answer

| Field | Effect |
|---|---|
| `weak_in_evaluation = yes` | Adds a warning to that class's candidate card |
| `evaluation_note` | The warning text shown with it |
| `limitations` | Extra limitation lines shown for matched classes |
| `next_step` | Replaces the default next step when this class matches best |

Mosaic Virus and Powdery Mildew should be `weak_in_evaluation = yes`, based on the offline results below.

### Test records

`data\seed\sample_test_records.csv` contains 8 records marked `source_id = DEV_TEST_NOT_FOR_USE`.
Use them to test the pipeline. Do not use them for real answers. Replace them with expert records as
soon as possible.

---

## 4. Frontend setup

Run these from the project root, in a separate terminal from the backend.

```cmd
cd "E:\Desktop-Copy\Arva Tech\plant-health-poc"
npm create vite@6 frontend -- --template react
cd frontend
npm install
```

Replace `frontend\src\App.jsx` and `frontend\src\index.css` with the versions in this repo.

### Start the frontend

```cmd
cd frontend
npm run dev
```

Open **http://localhost:5173**.

The backend allows requests from `localhost:5173` only (CORS). Use `localhost`, not `127.0.0.1`, for the frontend.

---

## 5. Running the full application

Use three terminals.

| Terminal | Folder | Command |
|---|---|---|
| 1. Backend | `backend` | `.venv\Scripts\activate` then `set HF_HOME=E:\hf_cache` then `uvicorn app.main:app --reload --port 8000` |
| 2. Frontend | `frontend` | `npm run dev` |
| 3. Testing (optional) | `backend` | curl commands below |

Then open http://localhost:5173, type a question, choose a photo, and click **Ask**.

Start the backend before the frontend. If the backend is down, the UI shows a request error.

---

## 6. API reference

Base URL: `http://127.0.0.1:8000`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, version, VLM provider, approved records, supported crops |
| GET | `/guidance/photos` | Capture guidance with reasons for each extra view |
| POST | `/cases` | Create a case (crop, region, duration, plant part, treatment, consent) |
| GET | `/cases` | List cases (queue summary) |
| GET | `/cases/{case_id}` | Case details, turns, feedback, reviews, summary |
| PATCH | `/cases/{case_id}` | Update optional case context |
| POST | `/cases/{case_id}/messages` | Send `text` and up to 4 `files`. Returns structured assessment |
| POST | `/cases/{case_id}/feedback` | Usefulness / dispute / next-action feedback |
| POST | `/cases/{case_id}/reviews` | Expert review |
| POST | `/cases/{case_id}/request-review` | Flag for expert queue |
| GET | `/reviews/queue` | Prioritised expert queue |
| GET | `/knowledge/records` | List records and sources |
| PATCH | `/knowledge/records/{id}` | Approve / retire / edit |
| GET | `/knowledge/coverage` | Coverage register |
| GET | `/knowledge/gaps` | Knowledge-gap log |
| GET | `/evidence/{id}` | Resolve cited evidence |
| GET | `/dashboard` | Latency, outcomes, gaps |
| POST | `/eval/run` | Pilot release gates |
| GET | `/images/{path}` | Dataset reference photo |
| GET | `/uploads/{case_id}/{name}` | Case upload preview |

### Example response (abridged)

```json
{
  "observations": ["1 image(s) received. Visual analysis not configured yet."],
  "candidate_causes": [{
    "name": "Little Leaf",
    "support": ["5 of 5 nearest reference photos", "mean similarity 0.89"],
    "against_or_missing": ["Visual similarity only. Not confirmed by inspection or lab test."],
    "evidence_ids": ["Raw_Dataset/Little_Leaf/ll_004.jpg"]
  }],
  "alternatives": [],
  "coverage": "partial",
  "limitations": ["Decision support only. ..."],
  "questions": [],
  "next_step": "Compare the photo with the cited reference photos and confirm with a local plant health expert.",
  "retrieved_evidence": [{"id": "Raw_Dataset/Little_Leaf/ll_004.jpg", "type": "image", "similarity": 0.92}],
  "policy_removals": 0,
  "model_version": "stub+clip-retrieval+keyword-0.3"
}
```

`coverage` is `none` (no approved match) or `partial`. It is never `strong` in this POC.

---

## 7. Testing

### 7.1 Quick checks

```cmd
curl.exe http://127.0.0.1:8000/health
```

Create a case. Put the JSON in `case.json` first, because Windows `curl` does not handle inline JSON well:

```json
{"crop": "brinjal", "region": "Sri Lanka", "symptom_duration": "about 2 weeks", "plant_part": "leaf"}
```

```cmd
curl.exe -X POST "http://127.0.0.1:8000/cases" -H "Content-Type: application/json" -d @case.json
```

Copy the `case_id` from the response, then:

**Text only**
```cmd
curl.exe -X POST "http://127.0.0.1:8000/cases/CASE_ID/messages" -F "text=brown spots on lower leaves"
```

**Photo + text**
```cmd
curl.exe -X POST "http://127.0.0.1:8000/cases/CASE_ID/messages" -F "text=brown spots on lower leaves" -F "files=@data\raw\sri_lanka_brinjal\Raw_Dataset\Little_Leaf\ll_006.jpg;type=image/jpeg" | python -m json.tool
```

Add `| python -m json.tool` to make the output readable.

### 7.2 Fair test photos (held-out, not in the reference index)

Use photos from `data\splits\main_test.csv`. These were never in the reference library, so they
cannot match themselves.

| Class | Example file | Expected top candidate |
|---|---|---|
| Little Leaf | `Raw_Dataset\Little_Leaf\ll_006.jpg` | Little Leaf |
| Healthy | `Raw_Dataset\Healthy\h_211.jpg` | Healthy |
| Mosaic Virus | `Raw_Dataset\Mosaic_Virus\mv_056.jpg` | Weak. Shows the Mosaic warning |
| Powdery Mildew | `Raw_Dataset\Powdery_Mildew\pm_056.jpg` | Weak. Shows the Powdery warning |

Do not test with photos from `main_index.csv` (the reference library). They match themselves at 1.0.

### 7.3 Reference photos in the browser

```
http://127.0.0.1:8000/images/Raw_Dataset/Little_Leaf/ll_004.jpg
```

Opens if the photo route works.

### 7.4 Offline retrieval evaluation

This measures the photo matcher alone, without the app.

```cmd
python scripts\evaluate_retrieval.py --dataset data\raw\sri_lanka_brinjal --splits data\splits
```

Results are saved to `data\results\`.

**Main split (grouped, stratified, 20% held out per class):** top-1 accuracy 66.1%
(95% CI 60.5% to 71.3%). Majority-class baseline 26.0%. Chance 12.5%.

Per class (top-1): Little Leaf 95%, Healthy 85%, Bacterial Leaf Spot 85%, Cercospora 74%,
Bacterial Blight 71%, Bacterial Wilt 63%, Powdery Mildew 40%, Mosaic Virus 39%.

**Leave-one-farm-out (stricter):** Only Healthy and Cercospora appear at more than one farm, so
other classes cannot be tested across farms. Healthy drops to 65% to 76% at new farms. Cercospora
drops to 37.5% when HORDI_Kandy (the only Wet Zone farm) is held out.

Run a farm holdout example:

```cmd
python scripts\evaluate_retrieval.py --dataset data\raw\sri_lanka_brinjal --splits data\splits --test-file data\splits\farm_HORDI_Kandy_test.csv --index-file data\splits\farm_HORDI_Kandy_index.csv
```

Use `--index all` to include augmented copies (robustness check). Use `--dry-run` to test the
pipeline with random vectors (scores are meaningless).

---

## 8. Configuration

Set in `backend/app/config.py`, or with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PH_DATA_DIR` | `backend/data` | Where all data files live |
| `PH_VLM_PROVIDER` | `stub` | VLM provider (only `stub` exists) |
| `HF_HOME` | Hugging Face default (C:) | Set to `E:\hf_cache` on Windows to save space |

Constants: `CROP = "brinjal"`, `TOP_K = 5`, `MAX_IMAGES_PER_MESSAGE = 4`.

---

## 9. Data, privacy, and git

- Dataset, uploads, embeddings, splits, results, the SQLite database, and `records.json` are
  git-ignored (see `.gitignore`).
- Commit: code, `records.example.json`, `expert_records_template.csv`, `dataset_sources.md`, and
  scripts.
- Do not commit `sample_test_records.csv` as if it were real. It is for testing only.
- Do not commit real expert records until they have been reviewed and signed off.
- Case data (farmer questions and photos) is stored locally in `poc.sqlite3` and `data/uploads/`.
  Do not share these without consent.

### Push to GitHub

```cmd
cd "E:\Desktop-Copy\Arva Tech\plant-health-poc"
git status
git add .
git commit -m "Describe the change"
git push origin main
```

Check `git status` first. If dataset or `.venv` files appear, the `.gitignore` is not applied.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No space left on device` during model download | CLIP cache is on C: | `set HF_HOME=E:\hf_cache` before starting uvicorn, then restart |
| `No space left on device` during pip install | Temp files on C: | Set `TMP`, `TEMP`, `PIP_CACHE_DIR` to a drive with space (see section 2) |
| `Could not find module numpy` / `No module named ...` | Packages installed in another environment | Activate `.venv`, then reinstall |
| `404 case not found` after restart | Cases were in the old in-memory store | Cases now persist in SQLite. Create a new case if the ID is unknown |
| `422 Expected UploadFile` in Swagger | Swagger's file field sends an empty value | Use curl for photo uploads |
| `approved_records: 0` | No approved expert records | Run `records_from_csv.py`, then restart uvicorn |
| Photo gives "no approved match" | Class has no approved record, or index not loaded | Check `/health`, check `records.json`, check `main_index.csv` exists |
| `embedding cache not found` | Embeddings not generated | Run the evaluator once (it creates `data\embeddings\ViT-B-32.npz`) |
| Images do not show in the UI | Image route not included, or wrong path | Check `main.py` includes the image router. Test `/images/...` in the browser |
| Frontend cannot reach backend | Backend not running, or wrong host | Start uvicorn on port 8000. Use `localhost:5173` for the frontend |
| Random `/profile` or `GET /` 404s in the log | Something outside this project calling localhost:8000 | Harmless. Ignore, or find the source |
| `npx create-vite` fails with ENOENT | Corrupt npx cache | Delete the half-created folder, then use `npm create vite@6` |

---

## 11. Known limitations

- **No VLM.** Visual observations are placeholders. Answers rely on photo matching and keyword matching.
- **Photo matching is appearance-based.** Similar-looking classes can match each other. Mosaic Virus
  and Powdery Mildew score poorly offline.
- **Farm bias.** Most classes come from one farm, so the main split overstates generalisation.
- **Small test sets.** Some classes have 20 to 35 test photos. Report confidence intervals.
- **Possible mislabel.** `Raw_Dataset\Powdery_Mildew\mv_006.jpg` has a Mosaic Virus prefix. Confirm with the dataset source.
- **Text matching is keyword only.** Semantic matching is not built.
- **Restart needed for approvals.** Records load at startup, so restart uvicorn after changing `records.json`.
- **No authentication.** Local POC only. Do not expose the server to the internet.
- **Safety filter list is a starting point.** `policy.py` covers a short list of chemical names and dose patterns.

---

## 12. Roadmap (after this MVP)

1. Expert-approved seed records for 2–3 crops (blocker for field pilot)
2. Frozen evaluation set (200–300 reviewed cases) and baseline comparison (VLM-only vs retrieval)
3. Calibrated diagnostic confidence (only after expert-labelled cases)
4. Auth / roles (user, reviewer, admin) and retention/deletion policy
5. PostgreSQL + object storage if scale requires it
6. Optional trusted-source web retrieval after core pipeline is stable