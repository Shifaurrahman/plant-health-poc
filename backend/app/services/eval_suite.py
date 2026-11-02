"""Lightweight evaluation endpoint for the pilot gates.

Diagnostic accuracy targets are set with the plant-health lead after a baseline.
This runner checks response contract gates on either stored eval fixtures or
the current knowledge/index configuration.
"""
import json

from app.config import EVAL_FILE, SUPPORTED_CROPS
from app.store import store


def _gate_records() -> dict:
    approved = store.approved_records()
    missing_source = [r["record_id"] for r in approved if not r.get("source_id")]
    todo = [
        r["record_id"] for r in approved
        if "TODO" in " ".join([
            str(r.get("symptoms") or ""),
            str(r.get("management_general") or ""),
        ]).upper()
    ]
    return {
        "approved_records": len(approved),
        "records_missing_source_id": missing_source,
        "approved_records_with_todo": todo,
    }


def run_eval() -> dict:
    record_gates = _gate_records()
    fixtures = []
    if EVAL_FILE.exists():
        fixtures = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    results = []
    for case in fixtures:
        results.append({
            "id": case.get("id"),
            "expected_outcome": case.get("expected_outcome"),
            "notes": case.get("notes"),
            "status": "queued_for_expert_scoring",
        })

    return {
        "status": "ok",
        "supported_crops": list(SUPPORTED_CROPS),
        "fixture_count": len(fixtures),
        "fixtures": results,
        "gates": {
            "source_ids_on_approved_records": not record_gates["records_missing_source_id"],
            "no_todo_in_approved_records": not record_gates["approved_records_with_todo"],
            "coverage_labels_required": True,
            "diagnostic_targets": "Set with the plant-health lead after the baseline. Not claimed here.",
        },
        "record_audit": record_gates,
        "note": (
            "Keep evaluation examples out of the reference index. "
            "Do not treat this endpoint as a reliability claim."
        ),
    }
