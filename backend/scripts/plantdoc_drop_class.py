#!/usr/bin/env python3
"""Remove a class from the PlantDoc tomato split CSVs (no index photos to match against)."""
import csv
import sys
from pathlib import Path

DROP = "Tomato_two_spotted_spider_mites_leaf"

for name in ["plantdoc_tomato_index.csv", "plantdoc_tomato_test.csv"]:
    p = Path("data/splits") / name
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if r["class"] != DROP]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(name, "rows kept:", len(rows))