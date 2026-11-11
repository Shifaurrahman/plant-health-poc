# scripts/eval_multimodal.py
"""Multimodal retrieval eval on manual photos.

Index   = train + valid rows of manual_manifest.csv, embedded as image + record_text.
Queries = test rows, two modes:
  image_only : photo alone
  image_text : photo + one farmer-style query line (round-robin over query lines)
Metric: top-1 and top-5 label match per class (nearest index neighbours).
Classes with no test rows are skipped.

Saves query vectors to data/embeddings/manual_query_vectors.npz for eval_text_baseline.py.

Usage (run from backend/):
    python -m scripts.eval_multimodal             # uses index cache if present
    python -m scripts.eval_multimodal --rebuild   # re-embed the index
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.services.embedder import MODEL, embed_multimodal

BACKEND = Path(__file__).resolve().parent.parent
MANIFEST = BACKEND / "data" / "splits" / "manual_manifest.csv"
CACHE = BACKEND / "data" / "embeddings" / f"manual_multimodal_{Path(MODEL).name.lower()}.npz"
QUERY_CACHE = BACKEND / "data" / "embeddings" / "manual_query_vectors.npz"
BATCH = 8


def load_manifest() -> list[dict]:
    with MANIFEST.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def embed_in_batches(items: list[dict]) -> np.ndarray:
    out = []
    for i in range(0, len(items), BATCH):
        out.append(embed_multimodal(items[i:i + BATCH]))
        print(f"  {min(i + BATCH, len(items))}/{len(items)}", end="\r")
    print()
    return np.concatenate(out)


def query_text_for(row: dict, k: int) -> str:
    lines = [t for t in row["query_texts"].split("||") if t]
    return lines[k % len(lines)] if lines else ""


def build_or_load_index(rows: list[dict]):
    index_rows = [r for r in rows if r["split"] in ("train", "valid")]
    paths = np.array([r["image_path"] for r in index_rows])
    labels = np.array([r["class"] for r in index_rows])

    if CACHE.exists() and "--rebuild" not in sys.argv:
        z = np.load(CACHE)
        if len(z["paths"]) == len(paths) and (z["paths"] == paths).all():
            print("Loaded index cache:", CACHE)
            return z["vectors"], labels
        print("Index cache stale (manifest changed). Rebuilding.")

    print(f"Embedding index: {len(index_rows)} images")
    items = [{"image": r["image_path"], "text": r["record_text"]} for r in index_rows]
    vecs = embed_in_batches(items)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, vectors=vecs, labels=labels, paths=paths)
    print("Saved index cache:", CACHE, vecs.shape)
    return vecs, labels


def score(q_vecs, q_labels, idx_vecs, idx_labels) -> dict:
    sims = q_vecs @ idx_vecs.T
    top5 = idx_labels[np.argsort(-sims, axis=1)[:, :5]]
    hit1 = top5[:, 0] == q_labels
    hit5 = (top5 == q_labels[:, None]).any(axis=1)

    per_cls = defaultdict(lambda: [0, 0, 0])  # n, top1, top5
    for lbl, h1, h5 in zip(q_labels, hit1, hit5):
        per_cls[lbl][0] += 1
        per_cls[lbl][1] += int(h1)
        per_cls[lbl][2] += int(h5)
    return {
        "overall_top1": float(hit1.mean()),
        "overall_top5": float(hit5.mean()),
        "per_class": dict(per_cls),
    }


def print_report(name: str, res: dict) -> None:
    print(f"\n== {name} ==")
    print(f"overall  top1 {res['overall_top1']*100:5.1f}%   top5 {res['overall_top5']*100:5.1f}%")
    print(f"{'class':<30}{'n':>5}{'top1':>9}{'top5':>9}")
    for cls in sorted(res["per_class"]):
        n, t1, t5 = res["per_class"][cls]
        print(f"{cls:<30}{n:>5}{t1/n*100:>8.1f}%{t5/n*100:>8.1f}%")


def main() -> None:
    rows = load_manifest()
    test_rows = [r for r in rows if r["split"] == "test"]
    skipped = sorted({r["class"] for r in rows} - {r["class"] for r in test_rows})
    print(f"Test queries: {len(test_rows)}   skipped (no test rows): {skipped}")

    idx_vecs, idx_labels = build_or_load_index(rows)
    q_labels = np.array([r["class"] for r in test_rows])

    items_img = [{"image": r["image_path"]} for r in test_rows]
    items_txt = [{"image": r["image_path"], "text": query_text_for(r, k)}
                 for k, r in enumerate(test_rows)]

    print("Embedding queries: image_only")
    q_img = embed_in_batches(items_img)
    print("Embedding queries: image_text")
    q_txt = embed_in_batches(items_txt)

    print_report("image_only", score(q_img, q_labels, idx_vecs, idx_labels))
    print_report("image_text", score(q_txt, q_labels, idx_vecs, idx_labels))

    QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(QUERY_CACHE, image_only=q_img, image_text=q_txt)
    print("Saved query vectors:", QUERY_CACHE)


if __name__ == "__main__":
    main()