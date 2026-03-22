"""
data_pipeline/finalize_incidents.py

Option A — Transfer confirmed reviewed incidents from the Review tab into the
Incidents tab as clean, structured records ready for database migration.

Reads rows from the Review sheet where review_status == 'reviewed', strips all
pipeline-internal columns (pred_*, conf_*, uncertain_*, IRR columns, etc.), and
appends clean incident records to the Incidents tab. Rows whose row_id is already
present in the Incidents tab are skipped to prevent duplicates.

The Incidents tab is the publication-ready layer: one row = one confirmed incident,
with only the facts and labels — no model scores, no pipeline metadata.

Usage:
    # Preview what would be transferred (no writes):
    python data_pipeline/finalize_incidents.py --dry-run

    # Transfer all newly reviewed rows:
    python data_pipeline/finalize_incidents.py

    # Also transfer rows that are reviewed but have no human labels filled
    # (not recommended for training, but useful for archiving):
    python data_pipeline/finalize_incidents.py --allow-unlabeled
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.database.sheets_interface import (  # noqa: E402
    get_client,
    SPREADSHEET_ID,
    REVIEW_SHEET_NAME,
    INCIDENTS_SHEET_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label columns — the 10 binary violence labels
# ---------------------------------------------------------------------------
LABELS = [
    "raid", "arrest_detention", "physical_assault", "coercive_actions",
    "restriction_of_freedoms", "religious_encroachment", "harm_to_property",
    "dispossession", "multi_community_incident", "protest",
]

# Columns to keep in the Incidents tab (matches actual Incidents sheet headers)
INCIDENT_COLS = [
    "incident_id",      # generated here from row_id
    "date",
    "governorate",
    "area",
    "location_name",
    "location_type",
    "perpetrator",
    "description",
] + LABELS + [
    "review_status",
    "reviewer_notes",
    "source",           # primary source URL
    "url",
]

# Review tab columns that are pipeline-internal and should NOT be carried over
_PIPELINE_PREFIXES = ("pred_", "conf_", "uncertain_", "human2_", "double_review")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sheet(sh, name: str):
    try:
        return sh.worksheet(name)
    except Exception as e:
        log.error("Could not open sheet '%s': %s", name, e)
        sys.exit(1)


def _is_pipeline_col(col: str) -> bool:
    return any(col.startswith(p) for p in _PIPELINE_PREFIXES)


def _resolve_label(review_row: dict, label: str) -> str:
    """
    Use the human label if filled, otherwise fall back to the model prediction.
    Returns '1', '0', or '' (empty = unknown).
    """
    human = str(review_row.get(f"human_{label}", "")).strip()
    if human in {"0", "1"}:
        return human
    pred = str(review_row.get(f"pred_{label}", "")).strip()
    if pred in {"0", "1"}:
        return pred
    return ""


def _row_id_to_incident_id(row_id: str) -> str:
    """
    Derive a stable incident_id from a review row_id.
    row_id format: <batch_ts>-<seq>  e.g. '20260322_051736-042'
    incident_id format: REVIEW-<row_id>
    """
    return f"REVIEW-{row_id}"


# ---------------------------------------------------------------------------
# Core transfer logic
# ---------------------------------------------------------------------------

def transfer(dry_run: bool = False, allow_unlabeled: bool = False) -> None:
    client = get_client()
    sh = client.open_by_key(SPREADSHEET_ID)

    review_ws   = _get_sheet(sh, REVIEW_SHEET_NAME)
    incident_ws = _get_sheet(sh, INCIDENTS_SHEET_NAME)

    # ── Read Review tab ──────────────────────────────────────────────────────
    review_vals = review_ws.get_all_values()
    if len(review_vals) < 2:
        log.info("Review tab is empty — nothing to transfer.")
        return

    review_headers = review_vals[0]
    review_rows = [dict(zip(review_headers, row)) for row in review_vals[1:]]

    reviewed = [
        r for r in review_rows
        if r.get("review_status", "").strip().lower() == "reviewed"
    ]
    log.info("Review tab: %d total rows, %d with review_status='reviewed'",
             len(review_rows), len(reviewed))

    # ── Read existing Incidents tab ──────────────────────────────────────────
    inc_vals = incident_ws.get_all_values()
    inc_headers = inc_vals[0] if inc_vals else []

    # Collect already-present incident_ids to avoid duplicates
    if inc_vals and len(inc_vals) > 1:
        id_col = inc_headers.index("incident_id") if "incident_id" in inc_headers else 0
        existing_ids = {row[id_col] for row in inc_vals[1:] if len(row) > id_col}
    else:
        existing_ids = set()

    log.info("Incidents tab: %d existing records", len(existing_ids))

    # ── Build rows to append ─────────────────────────────────────────────────
    rows_to_add: list[list] = []
    skipped_duplicate = 0
    skipped_unlabeled = 0

    for r in reviewed:
        row_id     = r.get("row_id", "").strip()
        inc_id     = _row_id_to_incident_id(row_id) if row_id else ""

        if inc_id in existing_ids:
            skipped_duplicate += 1
            continue

        # Check label coverage
        label_vals = [_resolve_label(r, lbl) for lbl in LABELS]
        filled = sum(1 for v in label_vals if v in {"0", "1"})
        if not allow_unlabeled and filled == 0:
            skipped_unlabeled += 1
            continue

        # Build clean incident record
        source_url = r.get("source_1", r.get("source", "")).strip()
        data = {
            "incident_id":   inc_id,
            "date":          r.get("date", "").strip(),
            "governorate":   r.get("governorate", "").strip(),
            "area":          r.get("area", "").strip(),
            "location_name": r.get("location_name", "").strip(),
            "location_type": r.get("location_type", "").strip(),
            "perpetrator":   r.get("perpetrator", "").strip(),
            "description":   r.get("description", "").strip(),
            "review_status": "reviewed",
            "reviewer_notes": r.get("reviewer_notes", "").strip(),
            "source":        source_url,
            "url":           source_url,
        }
        for lbl, val in zip(LABELS, label_vals):
            data[lbl] = val

        # Align to Incidents sheet column order
        row_data = [data.get(h, "") for h in inc_headers] if inc_headers else [
            data.get(c, "") for c in INCIDENT_COLS
        ]
        rows_to_add.append(row_data)
        existing_ids.add(inc_id)  # prevent same-batch duplicates

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 56)
    print("  FINALIZE INCIDENTS — TRANSFER SUMMARY")
    print("=" * 56)
    print(f"  Reviewed rows found      : {len(reviewed)}")
    print(f"  Already in Incidents tab : {skipped_duplicate}")
    print(f"  Skipped (no labels)      : {skipped_unlabeled}")
    print(f"  Rows to transfer         : {len(rows_to_add)}")
    print("=" * 56)
    print()

    if not rows_to_add:
        log.info("Nothing new to transfer.")
        return

    if dry_run:
        log.info("--dry-run: no rows written.")
        for i, row in enumerate(rows_to_add[:5]):
            print(f"  Preview [{i+1}]: {dict(zip(inc_headers or INCIDENT_COLS, row))}")
        return

    incident_ws.append_rows(rows_to_add)
    log.info("Transferred %d rows to Incidents tab.", len(rows_to_add))
    print(f"Done. {len(rows_to_add)} incident(s) added to the Incidents tab.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer reviewed incidents from the Review tab to the Incidents tab."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be transferred without writing anything.",
    )
    parser.add_argument(
        "--allow-unlabeled", action="store_true",
        help="Include rows with no human labels filled (uses model predictions as fallback).",
    )
    args = parser.parse_args()
    transfer(dry_run=args.dry_run, allow_unlabeled=args.allow_unlabeled)


if __name__ == "__main__":
    main()
