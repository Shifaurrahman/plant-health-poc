#!/usr/bin/env python3
"""Build evaluation splits from metadata.csv.

MAIN split (grouped, stratified):
  Per class, hold out ~20% of source photos for testing. All copies of a
  photo stay together. Test queries = Raw originals of held-out photos.
  Index = every other image, all phases and copies. Fixed seed (42).

SECONDARY split (leave-one-farm-out):
  For each farm F: test = Raw originals from F; index = images from other farms.
  Report only for classes present at more than one farm.

Usage (run from backend/):
    python scripts/make_splits.py data/raw/sri_lanka_brinjal/metadata.csv data/splits
"""
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ["relative_path", "class", "collection_site", "phase", "derivation", "source_image"]
TEST_FRACTION = 0.2
SEED = 42


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def check_one_site_per_photo(rows: list[dict]) -> None:
    sites = defaultdict(set)
    for r in rows:
        sites[r["source_image"]].add(r["collection_site"])
    bad = {k: v for k, v in sites.items() if len(v) > 1}
    if bad:
        sys.exit(f"ERROR: {len(bad)} photos appear at more than one farm, e.g. {next(iter(bad.items()))}")


def make_main_split(rows: list[dict], out: Path) -> None:
    raw = [r for r in rows if r["phase"] == "Raw" and r["derivation"] == "original"]
    rng = random.Random(SEED)

    test_sources: set[str] = set()
    for cls in sorted({r["class"] for r in raw}):
        sources = sorted({r["source_image"] for r in raw if r["class"] == cls})
        rng.shuffle(sources)
        k = max(1, round(len(sources) * TEST_FRACTION))
        test_sources.update(sources[:k])

    test = [r for r in raw if r["source_image"] in test_sources]
    index = [r for r in rows if r["source_image"] not in test_sources]
    assert not any(r["source_image"] in test_sources for r in index), "leak: test photo in index"

    write(out / "main_test.csv", test)
    write(out / "main_index.csv", index)

    print(f"MAIN split: {len(test)} test photos, {len(index)} index images")
    test_counts = Counter(r["class"] for r in test)
    index_counts = Counter(r["class"] for r in index)
    for cls in sorted(test_counts):
        print(f"  {cls:<25} test {test_counts[cls]:>4}   index images {index_counts[cls]:>5}")


def make_farm_splits(rows: list[dict], out: Path) -> None:
    check_one_site_per_photo(rows)
    for farm in sorted({r["collection_site"] for r in rows}):
        test = [r for r in rows
                if r["collection_site"] == farm and r["phase"] == "Raw" and r["derivation"] == "original"]
        index = [r for r in rows if r["collection_site"] != farm]
        test_sources = {r["source_image"] for r in test}
        assert not any(r["source_image"] in test_sources for r in index), "leak: test photo in index"

        write(out / f"farm_{farm}_test.csv", test)
        write(out / f"farm_{farm}_index.csv", index)

        print(f"\nFARM holdout {farm}: {len(test)} test photos, {len(index)} index images")
        for cls, n in sorted(Counter(r["class"] for r in test).items()):
            print(f"  {n:>4}  {cls}")


def main(meta_path: str, out_dir: str) -> None:
    rows = load(meta_path)
    out = Path(out_dir)
    make_main_split(rows, out)
    make_farm_splits(rows, out)
    print(f"\nWrote splits to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])