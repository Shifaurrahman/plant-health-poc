#!/usr/bin/env python3
"""Convert the expert's CSV into data/seed/records.json, with validation.

Usage (run from backend/):
    python scripts/records_from_csv.py data/seed/expert_records.csv data/seed/records.json

Rules:
- Every row needs the required columns (see TEMPLATE header).
- condition must be one of the 8 brinjal classes.
- review_status: pending or approved. approved needs reviewed_by and reviewed_on,
  and must not contain "TODO".
- List fields (alternatives, limitations) use ";" as separator.
- If any row has an error, nothing is written and the script exits with code 1.
Warnings (for example a chemical name in management_general) do not block the write.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so we can import app.services.policy
from app.services.policy import CHEMICAL_TERMS  # noqa: E402

CLASSES = {"Bacterial Blight", "Bacterial Leaf Spot", "Bacterial Wilt", "Cercospora Leaf Spot",
           "Healthy", "Little Leaf", "Mosaic Virus", "Powdery Mildew"}
CATEGORIES = {"disease", "pest", "abiotic", "healthy"}
STATUSES = {"pending", "approved", "retired"}
REQUIRED_COLUMNS = ["record_id", "condition", "category", "symptoms", "distinguishing_features",
                    "management_general", "source_id", "review_status"]


def split_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(";") if x.strip()]


def is_yes(value: str) -> bool:
    return value.strip().lower() in {"yes", "true", "1", "y"}


def row_to_record(row: dict) -> dict:
    return {
        "record_id": row["record_id"].strip(),
        "crop": "brinjal",
        "condition_id": row["condition"].strip().lower().replace(" ", "_"),
        "condition": row["condition"].strip(),
        "category": row["category"].strip().lower(),
        "affected_organs": split_list(row.get("affected_organs", "")),
        "label_basis": row.get("label_basis", "").strip() or "expert_suspected",
        "publication_status": row["review_status"].strip().lower(),
        "version": "1",
        "synonyms": [],
        "symptoms": row["symptoms"].strip(),
        "distinguishing_features": row["distinguishing_features"].strip(),
        "alternatives": split_list(row.get("alternatives", "")),
        "management_general": row["management_general"].strip(),
        "source_id": row["source_id"].strip(),
        "review_status": row["review_status"].strip().lower(),
        "reviewed_by": row.get("reviewed_by", "").strip() or None,
        "reviewed_on": row.get("reviewed_on", "").strip() or None,
        "weak_in_evaluation": is_yes(row.get("weak_in_evaluation", "")),
        "evaluation_note": row.get("evaluation_note", "").strip() or None,
        "limitations": split_list(row.get("limitations", "")),
        "next_step": row.get("next_step", "").strip() or None,
        "images": [],
    }


def check(rec: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for field in REQUIRED_COLUMNS:
        if not rec.get(field):
            errors.append(f"missing {field}")
    if rec["condition"] and rec["condition"] not in CLASSES:
        errors.append(f"condition '{rec['condition']}' is not one of the 8 classes")
    if rec["category"] and rec["category"] not in CATEGORIES:
        errors.append(f"category must be one of {sorted(CATEGORIES)}")
    if rec["review_status"] and rec["review_status"] not in STATUSES:
        errors.append("review_status must be 'pending' or 'approved'")
    if rec["review_status"] == "approved":
        if not rec["reviewed_by"]:
            errors.append("approved record needs reviewed_by")
        if not rec["reviewed_on"]:
            errors.append("approved record needs reviewed_on")
        text = " ".join([rec["symptoms"], rec["distinguishing_features"], rec["management_general"]])
        if "TODO" in text.upper():
            errors.append("approved record still contains TODO text")
    if rec["weak_in_evaluation"] and not rec["evaluation_note"]:
        warnings.append("weak_in_evaluation is yes but evaluation_note is empty")
    for term in CHEMICAL_TERMS:
        if term in rec["management_general"].lower():
            warnings.append(f"chemical name '{term}' in management_general: confirm it is in the source")
    return errors, warnings


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    with src.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            sys.exit(f"CSV is missing columns: {missing}")
        rows = list(reader)

    records, seen_ids, total_errors = [], set(), 0
    for line_no, row in enumerate(rows, start=2):  # line 1 is the header
        rec = row_to_record(row)
        errors, warnings = check(rec)
        if rec["record_id"] in seen_ids:
            errors.append("duplicate record_id")
        seen_ids.add(rec["record_id"])
        for e in errors:
            print(f"ERROR  line {line_no} ({rec['record_id'] or '-'}): {e}")
        for w in warnings:
            print(f"WARN   line {line_no} ({rec['record_id'] or '-'}): {w}")
        total_errors += len(errors)
        records.append(rec)

    if total_errors:
        print(f"\n{total_errors} error(s). Nothing written. Fix the CSV and run again.")
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(__import__("json").dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    approved = sum(r["review_status"] == "approved" for r in records)
    print(f"Wrote {len(records)} records to {dst} (approved: {approved}, pending: {len(records) - approved}).")


if __name__ == "__main__":
    main()