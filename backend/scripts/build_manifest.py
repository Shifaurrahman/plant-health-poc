#!/usr/bin/env python3
"""Scan a class-folder dataset and write a manifest CSV (one row per image).

Usage:
    python scripts/build_manifest.py <dataset_dir> <out_csv>

Example:
    python scripts/build_manifest.py data/raw/sri_lanka_brinjal/Phase_II_Dataset data/raw/manifest_brinjal_phase2.csv

Only lists files. It does not approve records or change any data.
"""
import csv
import sys
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main(dataset_dir: str, out_csv: str) -> None:
    root = Path(dataset_dir)
    if not root.is_dir():
        sys.exit(f"not a folder: {root}")

    rows = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = cls_dir.name.replace("_", " ")  # Bacterial_Leaf_Spot -> Bacterial Leaf Spot
        for img in sorted(cls_dir.rglob("*")):
            if img.is_file() and img.suffix.lower() in IMAGE_EXT:
                rows.append({"image_path": img.relative_to(root).as_posix(), "class_label": label})

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "class_label"])
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["class_label"]] = counts.get(r["class_label"], 0) + 1
    for label in sorted(counts):
        print(f"{counts[label]:>6}  {label}")
    print(f"total: {len(rows)} images -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])