# scripts/new_build_manifest.py
"""Build manual_manifest.csv from disease_map.csv + manual photos.

Run from backend/:
    python scripts\\new_build_manifest.py

Split:
    train_/valid_/test_ prefixes keep their split.
    Files without a prefix (brinjal) get a seeded, per-class 70/30 train/test split.
"""
import csv
import random
from collections import Counter
from pathlib import Path

from dataset_map import IMG_DIR, IMG_EXT, load_map, description_lines

BACKEND = Path(__file__).resolve().parent.parent
OUT = BACKEND / "data" / "splits" / "manual_manifest.csv"
SEED = 42
TEST_FRACTION = 0.3
FIELDS = ["image_path", "crop", "disease", "is_healthy", "class",
          "split", "record_text", "query_texts"]


def split_of(name: str) -> str:
    for s in ("train", "valid", "test"):
        if name.startswith(s + "_"):
            return s
    return "unknown"


def assign_unknown_splits(rows: list[dict]) -> None:
    by_cls: dict[str, list[dict]] = {}
    for r in rows:
        if r["split"] == "unknown":
            by_cls.setdefault(r["class"], []).append(r)

    rng = random.Random(SEED)
    for cls in sorted(by_cls):
        items = sorted(by_cls[cls], key=lambda r: r["image_path"])
        rng.shuffle(items)
        n_test = max(1, round(len(items) * TEST_FRACTION))
        for i, r in enumerate(items):
            r["split"] = "test" if i < n_test else "train"


def main() -> None:
    print("Image dir:", IMG_DIR, "exists:", IMG_DIR.exists())
    mp = load_map()
    print("Map rows:", len(mp))

    rows: list[dict] = []
    skipped: set[str] = set()
    desc_cache: dict[str, list[str]] = {}

    for img in sorted(IMG_DIR.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMG_EXT:
            continue
        folder = img.parent.name
        row = mp.get(folder)
        if row is None:
            skipped.add(folder)
            continue
        if folder not in desc_cache:
            desc_cache[folder] = description_lines(row)
        lines = desc_cache[folder]
        rows.append({
            "image_path": str(img),
            "crop": row["crop"],
            "disease": row["disease_name"],
            "is_healthy": row["is_healthy"],
            "class": folder,
            "split": split_of(img.name),
            "record_text": lines[0],
            "query_texts": "||".join(lines[1:]),
        })

    if skipped:
        print("WARN folders not in disease_map.csv (skipped):", sorted(skipped))

    assign_unknown_splits(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows -> {OUT}")
    for k, v in sorted(Counter((r["crop"], r["class"], r["split"]) for r in rows).items()):
        print(k, v)


if __name__ == "__main__":
    main()