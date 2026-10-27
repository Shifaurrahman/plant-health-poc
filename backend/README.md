# Plant Health POC: backend

## Run
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Docs: http://localhost:8000/docs

## Test with curl
```bash
# 1. create a case
curl -X POST localhost:8000/cases -H "Content-Type: application/json" \
  -d '{"crop":"brinjal","region":"Sri Lanka"}'

# 2. send a question (and optionally images) to that case
curl -X POST localhost:8000/cases/<case_id>/messages \
  -F "text=brown spots on lower leaves" -F "files=@leaf.jpg"
```

## Adding knowledge
1. Copy `data/seed/records.example.json` to `data/seed/records.json`.
2. Fill in symptoms and distinguishing features from a reviewed source.
3. Set `review_status` to `approved` only after expert sign-off. Pending records never reach retrieval.
4. Restart the server.

## Status
- Working: case creation, image upload (type and count checks), keyword retrieval over approved records, evidence validation, unknown-coverage outcome.
- Stub: VLM (`PH_VLM_PROVIDER=stub`). Replace `app/services/vlm.py` with a real adapter.
- Not yet: multimodal embeddings, FAISS index, persistent storage, auth, file-size limits.

## Downloading datasets
The sandbox cannot reach Kaggle or Mendeley. Run the script on your own machine:
```bash
# Kaggle: one-time setup, get kaggle.json from kaggle.com > Settings > API
pip install kaggle
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Mendeley: copy the link from the download button in your browser (right-click > Copy link address)
BRINJAL_ZIP_URL="<paste link>" TEA_ZIP_URL="<paste link>" bash scripts/download_datasets.sh
```