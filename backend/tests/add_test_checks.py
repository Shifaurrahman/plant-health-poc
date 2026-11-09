# tests/add_test_checks.py
import json
from pathlib import Path

path = Path("data/seed/records.json")
records = json.loads(path.read_text(encoding="utf-8"))

DRAFT = {
    "Little Leaf": [
        "DRAFT TEST CHECK: look for short internodes on the shoot",
        "DRAFT TEST CHECK: check whether leaves are small and narrow across the whole plant",
    ],
    "Mosaic Virus": [
        "DRAFT TEST CHECK: look for mottling that spreads across the whole leaf",
        "DRAFT TEST CHECK: check the youngest leaves for distortion",
    ],
    "Cercospora Leaf Spot": [
        "DRAFT TEST CHECK: look for concentric rings inside spots on older leaves",
    ],
}

for rec in records:
    if rec["condition"] in DRAFT:
        rec["confirm_by"] = DRAFT[rec["condition"]]

path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
json.loads(path.read_text(encoding="utf-8"))
print("updated and valid:", [r["condition"] for r in records if r.get("confirm_by")])