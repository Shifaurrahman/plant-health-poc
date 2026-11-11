"""Text-only and image-based retrieval, with and without a crop filter.

Crop filter: a farmer selects the crop first, so a query may only match records
and photos from that crop. Without the filter, brinjal and tomato records compete.

Usage (run from backend/):
    python -m scripts.eval_text_baseline
"""
import csv
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from app.services.embedder import encode_document, encode_query

BACKEND = Path(__file__).resolve().parent.parent
MANIFEST = BACKEND / "data" / "splits" / "manual_manifest.csv"
QUERY_CACHE = BACKEND / "data" / "embeddings" / "manual_query_vectors.npz"
CACHE = BACKEND / "data" / "embeddings" / "manual_multimodal_embeddinggemma-2.npz"
PHASH_THRESHOLD = 6


def load_manifest():
    with MANIFEST.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def query_text_for(row, k):
    lines = [t for t in row["query_texts"].split("||") if t]
    return lines[k % len(lines)] if lines else ""


def report(name, q_vecs, q_labels, q_crops, idx_vecs, idx_labels, idx_crops, crop_filter):
    sims = q_vecs @ idx_vecs.T
    if crop_filter:
        same = q_crops[:, None] == idx_crops[None, :]
        sims = np.where(same, sims, -np.inf)
    top5 = idx_labels[np.argsort(-sims, axis=1)[:, :5]]
    hit1 = top5[:, 0] == q_labels
    hit5 = (top5 == q_labels[:, None]).any(axis=1)

    per = defaultdict(lambda: [0, 0, 0])
    for lbl, a, b in zip(q_labels, hit1, hit5):
        per[lbl][0] += 1
        per[lbl][1] += int(a)
        per[lbl][2] += int(b)

    tag = "crop-filtered" if crop_filter else "no crop filter"
    print(f"\n== {name} [{tag}] (n={len(q_labels)}) ==")
    print(f"overall  top1 {hit1.mean()*100:5.1f}%   top5 {hit5.mean()*100:5.1f}%")
    print(f"{'class':<32}{'n':>4}{'top1':>9}{'top5':>9}")
    for c in sorted(per):
        n, t1, t5 = per[c]
        print(f"{c:<32}{n:>4}{t1/n*100:>8.1f}%{t5/n*100:>8.1f}%")


def near_duplicate_paths(rows):
    index = [r for r in rows if r["split"] in ("train", "valid")]
    test = [r for r in rows if r["split"] == "test"]
    idx_h = [(imagehash.phash(Image.open(r["image_path"]).convert("RGB")), r["image_path"])
             for r in index]
    dups = set()
    for r in test:
        t = imagehash.phash(Image.open(r["image_path"]).convert("RGB"))
        d, _ = min((t - h, p) for h, p in idx_h)
        if d <= PHASH_THRESHOLD:
            dups.add(r["image_path"])
    return dups


def main():
    rows = load_manifest()
    index = [r for r in rows if r["split"] in ("train", "valid")]
    test = [r for r in rows if r["split"] == "test"]
    idx_labels = np.array([r["class"] for r in index])
    idx_crops = np.array([r["crop"] for r in index])
    q_labels = np.array([r["class"] for r in test])
    q_crops = np.array([r["crop"] for r in test])

    # text-only: one record_text per class (with its crop)
    uniq = {}
    for r in index:
        uniq.setdefault(r["class"], (r["crop"], r["record_text"]))
    u_labels = np.array(list(uniq.keys()))
    u_crops = np.array([v[0] for v in uniq.values()])
    u_text = encode_document([v[1] for v in uniq.values()])
    q_text = encode_query([query_text_for(r, k) for k, r in enumerate(test)])

    for flt in (False, True):
        report("text_only", q_text, q_labels, q_crops,
               u_text, u_labels, u_crops, crop_filter=flt)

    # image-based, near-duplicate test photos excluded
    dups = near_duplicate_paths(rows)
    keep = np.array([r["image_path"] not in dups for r in test])
    print(f"\nExcluded near-duplicate test photos: {len(dups)}")

    qz = np.load(QUERY_CACHE)
    idx_vecs = np.load(CACHE)["vectors"]
    for mode in ("image_only", "image_text"):
        for flt in (False, True):
            report(f"{mode} (no near-dups)", qz[mode][keep], q_labels[keep], q_crops[keep],
                   idx_vecs, idx_labels, idx_crops, crop_filter=flt)


if __name__ == "__main__":
    main()
