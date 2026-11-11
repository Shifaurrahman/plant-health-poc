# scripts/dataset_map.py
"""Single source of truth: image folder -> crop, disease, description.

Usage (run from backend/):
    python scripts\\dataset_map.py      # validates the map against the image folders
"""
import csv
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
IMG_DIR = BACKEND / "data" / "raw" / "manual" / "dataset"
DESC_DIR = BACKEND / "descriptions"
MAP_FILE = DESC_DIR / "disease_map.csv"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_map() -> dict[str, dict]:
    with MAP_FILE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    m: dict[str, dict] = {}
    for r in rows:
        if r["folder"] in m:
            raise SystemExit(f"duplicate folder in disease_map.csv: {r['folder']}")
        m[r["folder"]] = r
    return m


def description_lines(row: dict) -> list[str]:
    f = DESC_DIR / row["description_file"]
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def image_folders() -> set[str]:
    return {p.parent.name for p in IMG_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXT}


def validate() -> bool:
    m = load_map()
    folders = image_folders()
    ok = True

    no_row = sorted(folders - set(m))
    if no_row:
        ok = False
        print("ERROR image folder with no row in disease_map.csv:", no_row)

    unused = sorted(set(m) - folders)
    if unused:
        print("NOTE map rows with no image folder:", unused)

    for folder, row in sorted(m.items()):
        if not (DESC_DIR / row["description_file"]).exists():
            ok = False
            print("ERROR description file missing:", row["description_file"])
        elif folder in folders:
            print(f"OK  {row['crop']:<8} {folder:<38} -> {row['disease_name']}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    validate()