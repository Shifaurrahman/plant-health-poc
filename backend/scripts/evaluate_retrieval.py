#!/usr/bin/env python3
"""Retrieval-only evaluation on the main split.

Index  : reference images. Default "originals" = Raw-phase photos, excluding test photos.
         "all" = every non-test image, including augmented copies (robustness check).
Query  : held-out Raw originals from main_test.csv.
Predict: majority vote over the top-k most similar index images (cosine similarity).

Usage (run from backend/):
    # farm holdout (stricter): one run per farm
    python scripts/evaluate_retrieval.py --dataset data/raw/sri_lanka_brinjal --splits data/splits \
        --test-file data/splits/farm_HORDI_Kandy_test.csv --index-file data/splits/farm_HORDI_Kandy_index.csv
    python scripts/evaluate_retrieval.py --dataset data/raw/sri_lanka_brinjal --splits data/splits
    python scripts/evaluate_retrieval.py --dataset data/raw/sri_lanka_brinjal --splits data/splits --index all
    python scripts/evaluate_retrieval.py --dataset data/raw/sri_lanka_brinjal --splits data/splits --dry-run

First real run needs internet to download the CLIP weights (about 350 MB):
    pip install torch open_clip_torch pillow numpy

--dry-run uses random vectors. It checks the pipeline only and its scores mean nothing.
Embeddings are cached in data/embeddings/ so reruns are fast.
"""
import argparse
import csv
import json
import math
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def embed_clip(paths: list[str], dataset_root: Path, batch: int = 32) -> np.ndarray:
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = torch.stack([preprocess(Image.open(dataset_root / p).convert("RGB"))
                                for p in paths[i:i + batch]])
            f = model.encode_image(imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            print(f"  embedded {min(i + batch, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.concatenate(feats)


def embed_dry(paths: list[str]) -> np.ndarray:
    # Deterministic random vectors per path. For pipeline testing only.
    feats = []
    for p in paths:
        rng = np.random.default_rng(zlib.crc32(p.encode()))
        v = rng.standard_normal(64)
        feats.append(v / np.linalg.norm(v))
    return np.array(feats)


def get_embeddings(paths: list[str], dataset_root: Path, cache_file: Path, dry: bool) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        cache = dict(zip(data["paths"].tolist(), data["feats"]))
    missing = [p for p in paths if p not in cache]
    if missing:
        print(f"Embedding {len(missing)} new images ({'dry run' if dry else MODEL})")
        new = embed_dry(missing) if dry else embed_clip(missing, dataset_root)
        cache.update(zip(missing, new))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, paths=np.array(list(cache.keys())), feats=np.stack(list(cache.values())))
    return cache


def predict(query: np.ndarray, index: np.ndarray, labels: list[str], k: int) -> list[str]:
    sims = query @ index.T
    top = np.argsort(-sims, axis=1)[:, :k]
    preds = []
    for row in top:
        neighbour_labels = [labels[j] for j in row]
        counts = Counter(neighbour_labels)
        # Most votes wins. Ties go to the label that ranks higher (earlier in the neighbour list).
        winner = max(counts, key=lambda lab: (counts[lab], -neighbour_labels.index(lab)))
        preds.append(winner)
    return preds


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset root, e.g. data/raw/sri_lanka_brinjal")
    ap.add_argument("--splits", required=True, help="folder containing main_test.csv and main_index.csv")
    ap.add_argument("--index", choices=["originals", "all"], default="originals")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-file", help="default: <splits>/main_test.csv")
    ap.add_argument("--index-file", help="default: <splits>/main_index.csv")
    args = ap.parse_args()

    dataset_root = Path(args.dataset)
    splits = Path(args.splits)
    test_path = Path(args.test_file) if args.test_file else splits / "main_test.csv"
    index_path = Path(args.index_file) if args.index_file else splits / "main_index.csv"
    test = read_csv(test_path)
    index_rows = read_csv(index_path)
    split_tag = test_path.stem.replace("_test", "")  # main, or farm_<site>
    if args.index == "originals":
        index_rows = [r for r in index_rows if r["phase"] == "Raw"]

    classes = sorted({r["class"] for r in test})
    print(f"Index: {len(index_rows)} images ({args.index}). Test queries: {len(test)}. Classes: {len(classes)}")

    cache_file = Path("data/embeddings") / f"{'dryrun' if args.dry_run else MODEL}.npz"
    all_paths = sorted({r["relative_path"] for r in index_rows} | {r["relative_path"] for r in test})
    emb = get_embeddings(all_paths, dataset_root, cache_file, args.dry_run)

    idx_feats = np.stack([emb[r["relative_path"]] for r in index_rows])
    idx_labels = [r["class"] for r in index_rows]
    q_feats = np.stack([emb[r["relative_path"]] for r in test])
    truth = [r["class"] for r in test]

    results = {"index": args.index, "index_size": len(index_rows), "test_size": len(test),
               "model": "dry-run" if args.dry_run else f"{MODEL}/{PRETRAINED}", "runs": {}}
    for k in sorted({1, args.k}):
        preds = predict(q_feats, idx_feats, idx_labels, k)
        correct = sum(p == t for p, t in zip(preds, truth))
        lo, hi = wilson(correct, len(test))
        per_class = {}
        for cls in classes:
            idxs = [i for i, t in enumerate(truth) if t == cls]
            c = sum(preds[i] == cls for i in idxs)
            per_class[cls] = {"n": len(idxs), "correct": c, "acc": round(c / len(idxs), 3)}
        results["runs"][f"k={k}"] = {"accuracy": round(correct / len(test), 3),
                                     "ci95": [round(lo, 3), round(hi, 3)], "per_class": per_class}

    majority = max(Counter(truth).values()) / len(truth)
    results["baselines"] = {"majority_class": round(majority, 3), "chance": round(1 / len(classes), 3)}

    print(f"\nBaselines: majority-class {majority:.1%}, chance {1 / len(classes):.1%}")
    for name, run in results["runs"].items():
        print(f"\n{name}: accuracy {run['accuracy']:.1%}  (95% CI {run['ci95'][0]:.1%} to {run['ci95'][1]:.1%})")
        for cls, s in run["per_class"].items():
            print(f"  {cls:<25} {s['correct']:>3}/{s['n']:<3}  {s['acc']:.1%}")

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"retrieval_{split_tag}_{args.index}{'_dryrun' if args.dry_run else ''}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out_file}")


if __name__ == "__main__":
    sys.exit(main())