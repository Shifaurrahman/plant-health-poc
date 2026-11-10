#!/usr/bin/env python3
"""Grouped, stratified split for PlantDoc tomato crops.

Groups crops by source_image (one original photo = one group), holds out ~20% of
groups per class for test, and writes index and test CSVs with the same columns
as the brinjal splits. Fixed seed for reproducibility.

Usage (from backend/):
    python scripts/plantdoc_tomato_split.py data/raw/plantdoc_tomato/metadata.csv data/splits
"""
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

TEST_FRACTION = 0.2
SEED = 42
FIELDS = ["relative_path", "class", "source_image", "split_origin",
          "phase", "derivation", "collection_site"]


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(meta_path: str, out_dir: str) -> None:
    rows = load(Path(meta_path))
    out = Path(out_dir)
    rng = random.Random(SEED)

    # Each source photo is one group. Split groups per class, not crops.
    test_groups: set[str] = set()
    by_class: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_class[r["class"]].add(r["source_image"])
    for cls in sorted(by_class):
        groups = sorted(by_class[cls])
        rng.shuffle(groups)
        k = max(1, round(len(groups) * TEST_FRACTION))
        test_groups.update(groups[:k])

    test = [r for r in rows if r["source_image"] in test_groups]
    index = [r for r in rows if r["source_image"] not in test_groups]

    # Hard checks: no group on both sides.
    assert not ({r["source_image"] for r in test} & {r["source_image"] for r in index}), "leak"

    write(out / "plantdoc_tomato_test.csv", test)
    write(out / "plantdoc_tomato_index.csv", index)

    print(f"test: {len(test)} crops from {len({r['source_image'] for r in test})} source photos")
    print(f"index: {len(index)} crops from {len({r['source_image'] for r in index})} source photos")
    tc, ic = Counter(r["class"] for r in test), Counter(r["class"] for r in index)
    for cls in sorted(tc):
        print(f"  {cls:<42} test {tc[cls]:>4}   index {ic[cls]:>5}")
    print(f"wrote {out / 'plantdoc_tomato_index.csv'} and {out / 'plantdoc_tomato_test.csv'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])