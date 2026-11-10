#!/usr/bin/env python3
"""Crop tomato leaf boxes from PlantDoc (YOLO format) into class folders.

Keeps only tomato classes (19-27 in data.yaml). Each box becomes one image.
File names are short (<split>_<counter>.jpg) to avoid Windows path-length limits.
The original source image stem is kept in metadata.csv (source_image), for grouped splits.

Usage (from backend/):
    python scripts/plantdoc_crop_tomato.py "C:\\Users\\USER\\Downloads\\archive" data/raw/plantdoc_tomato
"""
import csv
import sys
from pathlib import Path

from PIL import Image

TOMATO_IDS = set(range(19, 28))
FALLBACK_NAMES = {
    19: "Tomato_Early_blight_leaf",
    20: "Tomato_Septoria_leaf_spot",
    21: "Tomato_leaf_bacterial_spot",
    22: "Tomato_leaf_late_blight",
    23: "Tomato_leaf_mosaic_virus",
    24: "Tomato_leaf_yellow_virus",
    25: "Tomato_leaf",
    26: "Tomato_mold_leaf",
    27: "Tomato_two_spotted_spider_mites_leaf",
}
SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
MIN_SIDE = 64  # skip tiny boxes
META_FIELDS = ["relative_path", "class", "source_image", "split_origin",
               "phase", "derivation", "collection_site"]


def load_class_names(root: Path) -> dict[int, str]:
    """Read names from data.yaml so folder names match PlantDoc exactly. Falls back to FALLBACK_NAMES."""
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return FALLBACK_NAMES
    try:
        import yaml  # PyYAML
        names = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["names"]
    except Exception:
        return FALLBACK_NAMES
    result = {}
    for i in TOMATO_IDS:
        if i < len(names):
            result[i] = names[i].strip().replace(" ", "_")
    return result or FALLBACK_NAMES


def crop_split(root: Path, split: str, out: Path, names: dict[int, str],
               rows: list, counter: list[int]) -> int:
    img_dir = root / split / "images"
    lab_dir = root / split / "labels"
    count = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lab_path = lab_dir / (img_path.stem + ".txt")
        if not lab_path.exists():
            continue
        img = None
        for line in lab_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if cls not in TOMATO_IDS:
                continue
            if img is None:
                img = Image.open(img_path).convert("RGB")
            W, H = img.size
            xc, yc, bw, bh = map(float, parts[1:])
            x1 = max(0, int((xc - bw / 2) * W))
            y1 = max(0, int((yc - bh / 2) * H))
            x2 = min(W, int((xc + bw / 2) * W))
            y2 = min(H, int((yc + bh / 2) * H))
            if (x2 - x1) < MIN_SIDE or (y2 - y1) < MIN_SIDE:
                continue
            cls_name = names[cls]
            cls_dir = out / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            name = f"{split}_{counter[0]:06d}.jpg"
            counter[0] += 1
            img.crop((x1, y1, x2, y2)).save(cls_dir / name, quality=95)
            rows.append({
                "relative_path": f"{out.name}/{cls_name}/{name}",
                "class": cls_name,
                "source_image": f"{split}__{img_path.stem}",
                "split_origin": split,
                "phase": "Raw",
                "derivation": "original",
                "collection_site": "PlantDoc",
            })
            count += 1
    return count


def main(root_str: str, out_str: str) -> None:
    root, out = Path(root_str), Path(out_str)
    out.mkdir(parents=True, exist_ok=True)
    names = load_class_names(root)
    rows: list = []
    counter = [0]
    total = 0
    for split in SPLITS:
        if not (root / split / "images").exists():
            print(f"skip {split}: no images folder")
            continue
        n = crop_split(root, split, out, names, rows, counter)
        print(f"{split}: {n} tomato boxes")
        total += n
    if not rows:
        sys.exit("No tomato boxes found. Check the archive path and the labels folders.")
    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"total: {total} crops -> {out}")
    print(f"metadata: {out / 'metadata.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])