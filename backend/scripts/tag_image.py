# scripts/tag_image.py
"""Show the class tag and matching description for an image, using disease_map.csv.

Usage (run from backend/):
    python scripts\\tag_image.py <image_path> [more paths...]
"""
import sys
from pathlib import Path

from dataset_map import load_map, description_lines


def tag_image(image_path: str, mp: dict) -> dict:
    p = Path(image_path)
    folder = p.parent.name                    # class = folder name
    row = mp.get(folder)
    if row is None:
        return {"image": p.name, "class": folder, "status": "NOT IN disease_map.csv"}
    lines = description_lines(row)
    return {
        "image": p.name,
        "class": folder,
        "crop": row["crop"],
        "disease": row["disease_name"],
        "description_file": row["description_file"],
        "description": lines[0],
        "status": "OK",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    mp = load_map()
    for arg in sys.argv[1:]:
        info = tag_image(arg, mp)
        for k, v in info.items():
            print(f"{k:<17}: {v}")
        print()