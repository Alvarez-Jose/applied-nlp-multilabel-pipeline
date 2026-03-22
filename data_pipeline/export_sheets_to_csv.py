"""
data_pipeline/export_sheets_to_csv.py

Exports the Incidents tab from Google Sheets into a normalized CSV that is
compatible with the training pipeline (data/processed/ format).

Position in the workflow:
    ingest_reports.py        ← scraper writes Reports tab
    [RA codes Incidents tab] ← human step
    export_sheets_to_csv.py  ← this script (pulls Incidents → CSV)
    export_training_data.py  ← reads processed CSVs, builds train/dev splits

Output:
    data/processed/incidents_from_sheets.csv

This file is in the same canonical schema as oppression_2023.csv etc.,
so export_training_data.py can include it without modification.

Usage:
    # Export all coded rows:
    python data_pipeline/export_sheets_to_csv.py

    # Export only rows marked reviewed:
    python data_pipeline/export_sheets_to_csv.py --reviewed-only

    # Require at least 2 label fields to be filled:
    python data_pipeline/export_sheets_to_csv.py --min-label-fields 2

    # Preview counts without writing:
    python data_pipeline/export_sheets_to_csv.py --dry-run

    # Custom output path:
    python data_pipeline/export_sheets_to_csv.py --output data/processed/my_export.csv

Requirements:
    pip install gspread google-auth pandas
    service_account.json must be in the project root with Sheets read access.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.database.sheets_interface import list_incidents  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Governorate code → full name (matches codebook.py)
# ---------------------------------------------------------------------------
GOVERNORATE_CODE_TO_NAME = {
    "NAB":  "Nablus",
    "QAL":  "Qalqilya",
    "TUB":  "Tubas",
    "SAL":  "Salfit",
    "TUL":  "Tulkarm",
    "JEN":  "Jenin",
    "JER":  "Jericho",
    "JVA":  "Jordan Valley",
    "RAM":  "Ramallah and al-Bireh",
    "BET":  "Bethlehem",
    "HEB":  "Hebron",
    "JERU": "Jerusalem",
}

# ---------------------------------------------------------------------------
# The 10 label columns — used for completeness checks
# ---------------------------------------------------------------------------
LABEL_COLS = [
    "raid",
    "arrest_detention",
    "physical_assault",
    "harm_to_property",
    "dispossession",
    "religious_encroachment",
    "restriction_of_freedoms",
    "coercive_actions",
    "protest",
    "multi_community_incident",
]

# ---------------------------------------------------------------------------
# Canonical output columns — must match normalize_raw.py CANONICAL_OUTPUT_COLS
# ---------------------------------------------------------------------------
CANONICAL_OUTPUT_COLS = [
    "row_id",
    "id", "date", "area", "location_name", "location_type", "governorate",
    "jurisdiction", "perpetrator", "palestinian_involvement",
    "raid", "arrest_detention", "number_arrested", "physical_assault",
    "killed", "number_injured", "coercive_actions", "restriction_of_freedoms",
    "restriction_of_freedoms_mechanisms", "religious_encroachment",
    "harm_to_property", "type_of_property_harmed", "dispossession",
    "type_of_property_dispossessed",
    "description", "multi_community_incident", "protest", "type_of_protest",
    "source_1", "source_2", "source_3", "source_4", "source_5", "source_6",
]


# ===========================================================================
# Column mapping
# ===========================================================================

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename Incidents sheet column names to the canonical schema used by normalize_raw.py.

    The Incidents sheet uses these column names (as of current setup):
        incident_id, date, governorate, area, location_name, location_type,
        perpetrator, description, <10 label cols>, review_status, reviewer_notes,
        source, url

    Only one rename is needed — incident_id → id. All other column names already
    match the canonical schema. The single 'source' column is mapped to 'source_1'.
    """
    renames = {
        "incident_id": "id",
        "source":      "source_1",   # sheet has single source col; canonical has source_1..6
    }
    df = df.rename(columns=renames)

    # Drop columns that have no place in the canonical training schema
    drop_cols = {
        "review_status", "reviewer_notes",   # workflow metadata, not features
        "incident_type", "material_loss",    # legacy cols that may appear in older exports
        "governorate_code",                  # not in current sheet, but guard against old data
    }
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Add any canonical columns absent from the sheet as empty strings
    for col in CANONICAL_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""

    return df


# ===========================================================================
# Row ID generation
# ===========================================================================

def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate row_id using the same logic as normalize_raw.py:
      {id}-{cumcount}   where cumcount handles duplicate IDs.

    Since incident_id in Sheets is already unique (enforced by create_incident),
    most rows will get suffix "-0".
    """
    id_key = df["id"].fillna("MISSING_ID").astype(str)
    suffix = df.groupby(id_key).cumcount().astype(str)
    df["row_id"] = (id_key + "-" + suffix).astype(str)

    if df["row_id"].duplicated().any():
        raise ValueError("row_id is not unique — check incident_id values in Sheets.")

    return df


# ===========================================================================
# Quality filter
# ===========================================================================

def is_ready(row: pd.Series, min_label_fields: int, reviewed_only: bool) -> tuple[bool, str]:
    """
    Return (keep: bool, reason: str) for a single row.

    A row is ready for training when:
      1. description is non-empty (≥ 20 characters)
      2. date is non-empty
      3. location_name or governorate is non-empty
      4. At least min_label_fields of the 10 labels are filled
      5. If reviewed_only: review_status == "reviewed"
    """
    desc = str(row.get("description", "")).strip()
    if len(desc) < 20:
        return False, f"description too short ({len(desc)} chars)"

    if not str(row.get("date", "")).strip():
        return False, "missing date"

    has_location = (
        bool(str(row.get("location_name", "")).strip())
        or bool(str(row.get("governorate", "")).strip())
    )
    if not has_location:
        return False, "missing location_name and governorate"

    filled_labels = sum(
        1 for col in LABEL_COLS
        if str(row.get(col, "")).strip() not in {"", "0", "false", "no"}
    )
    if filled_labels < min_label_fields:
        return False, f"only {filled_labels}/{len(LABEL_COLS)} label fields filled (min={min_label_fields})"

    if reviewed_only:
        status = str(row.get("review_status", "")).strip().lower()
        if status != "reviewed":
            return False, f"review_status='{status}' (need 'reviewed')"

    return True, "ok"


def filter_rows(
    df: pd.DataFrame,
    min_label_fields: int,
    reviewed_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into (ready, rejected).

    Returns both so the caller can log the rejection reasons.
    """
    mask = []
    reasons = []

    for _, row in df.iterrows():
        keep, reason = is_ready(row, min_label_fields, reviewed_only)
        mask.append(keep)
        reasons.append(reason)

    ready    = df[mask].copy()
    rejected = df[~pd.Series(mask)].copy()
    rejected["_reject_reason"] = [r for r, m in zip(reasons, mask) if not m]

    return ready, rejected


# ===========================================================================
# Main export logic
# ===========================================================================

def export(
    output_path: Path,
    min_label_fields: int,
    reviewed_only: bool,
    dry_run: bool,
) -> None:
    # --- 1. Pull Incidents from Sheets ---
    log.info("Connecting to Google Sheets and reading Incidents tab ...")
    try:
        rows = list_incidents()
    except Exception as e:
        log.error("Failed to read Incidents tab: %s", e)
        sys.exit(1)

    if not rows:
        log.warning("Incidents tab is empty — nothing to export.")
        return

    df_raw = pd.DataFrame(rows).fillna("")
    log.info("Pulled %d rows from Incidents tab.", len(df_raw))
    log.info("Sheets columns: %s", list(df_raw.columns))

    # --- 2. Map columns to canonical schema ---
    df = map_columns(df_raw)

    # --- 3. Quality filter ---
    ready, rejected = filter_rows(df, min_label_fields, reviewed_only)

    log.info(
        "Quality filter: %d rows ready, %d rows rejected.",
        len(ready), len(rejected),
    )

    if len(rejected):
        rejection_summary = rejected["_reject_reason"].value_counts()
        log.info("Rejection reasons:\n%s", rejection_summary.to_string())

    if len(ready) == 0:
        log.warning("No rows passed the quality filter. Nothing to write.")
        if reviewed_only:
            log.info("Tip: try running without --reviewed-only to see all rows.")
        return

    # --- 4. Add row_id ---
    ready = add_row_id(ready)

    # --- 5. Enforce canonical column order ---
    for col in CANONICAL_OUTPUT_COLS:
        if col not in ready.columns:
            ready[col] = ""
    ready = ready[CANONICAL_OUTPUT_COLS]

    # --- 6. Summary ---
    print("\n" + "=" * 58)
    print("  EXPORT SUMMARY")
    print("=" * 58)
    print(f"  Total rows in Sheets   : {len(df_raw)}")
    print(f"  Rows passing filter    : {len(ready)}")
    print(f"  Rows rejected          : {len(rejected)}")
    print()

    label_fill_rates = {
        col: (ready[col].astype(str).str.strip().isin({"", "0", "false", "no"}) == False).mean()  # noqa: E712
        for col in LABEL_COLS
        if col in ready.columns
    }
    print("  Label fill rates (ready rows):")
    for col, rate in label_fill_rates.items():
        print(f"    {col:35s} {rate*100:5.1f}%")

    print()
    print(f"  Output: {output_path}")
    print("=" * 58 + "\n")

    if dry_run:
        log.info("--dry-run: no file written.")
        return

    # --- 7. Write ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ready.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Wrote %d rows to %s", len(ready), output_path)


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export coded incidents from Google Sheets to a training-ready CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        default=str(PROJECT_ROOT / "data" / "processed" / "incidents_from_sheets.csv"),
        help="Output CSV path. Default: data/processed/incidents_from_sheets.csv",
    )
    parser.add_argument(
        "--min-label-fields",
        type=int,
        default=1,
        help=(
            "Minimum number of the 10 label fields that must be filled "
            "for a row to be included. Default: 1."
        ),
    )
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help=(
            "Only export rows where the 'review_status' column equals 'reviewed'. "
            "Use this when your Incidents sheet has a review workflow."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the summary but do not write the output file.",
    )
    args = parser.parse_args()

    export(
        output_path=Path(args.output),
        min_label_fields=args.min_label_fields,
        reviewed_only=args.reviewed_only,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
