"""
modeling/merge_reviewed_labels.py

Ingests a reviewed Excel/CSV sheet and converts human corrections into
finalized labels, ready to append to the master training dataset.

Workflow position:
    predict_and_prepare_review.py  →  [RA reviews in Excel]  →  merge_reviewed_labels.py

The script handles two input formats automatically:
    Full review sheet   (review_{ts}.csv / .xlsx)
        Contains pred_* columns — predictions are read directly.
    Compact review sheet (review_compact_{ts}.csv / .xlsx)
        Lacks pred_* columns — requires --predictions <predictions_{ts}.csv>
        to supply the model's per-label predictions.

Label resolution rule (per label, per row):
    If review_status == "reviewed" AND human_{label} is non-empty
        → use human label  (accepted values: 1/0, yes/no, true/false, y/n, x)
    Otherwise
        → fall back to model prediction

Usage:
    # Full review sheet (predictions built-in):
    python modeling/merge_reviewed_labels.py \\
        --reviewed data/reviewed/reviewed_20260321_124159.xlsx \\
        --output   data/reviewed/reviewed_incidents_20260321.csv \\
        --append-to data/training/master_reviewed_dataset.csv

    # Compact review sheet + predictions audit file:
    python modeling/merge_reviewed_labels.py \\
        --reviewed     data/reviewed/review_compact_20260321_124159.xlsx \\
        --predictions  data/review_exports/predictions_20260321_124159.csv \\
        --output       data/reviewed/reviewed_incidents_20260321.csv \\
        --append-to    data/training/master_reviewed_dataset.csv

Output columns (compatible with modeling/training/data/train.csv schema):
    Metadata    : row_id, id, date, year, source_file, area, location_name,
                  location_type, governorate, jurisdiction, perpetrator, description
    Subtypes    : physical_assault, harm_to_property, dispossession,
                  coercive_actions, restriction_of_freedoms  (preserved from input)
    Labels      : raid_y, arrest_detention_y, physical_assault_y,
                  harm_to_property_y, dispossession_y, religious_encroachment_y,
                  restriction_of_freedoms_y, coercive_actions_y,
                  protest_y, multi_community_incident_y
    Audit       : label_source_{label} (human/model × 10), human_override_count,
                  review_status, reviewed_by, reviewer_notes,
                  model_used, run_timestamp

Requirements:
    pip install pandas numpy openpyxl
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants — must match predict_and_prepare_review.py
# ---------------------------------------------------------------------------

LABELS: List[str] = [
    "raid_y",
    "arrest_detention_y",
    "physical_assault_y",
    "harm_to_property_y",
    "dispossession_y",
    "religious_encroachment_y",
    "restriction_of_freedoms_y",
    "coercive_actions_y",
    "protest_y",
    "multi_community_incident_y",
]

LABEL_SHORT: Dict[str, str] = {
    "raid_y":                      "raid",
    "arrest_detention_y":          "arrest_detention",
    "physical_assault_y":          "physical_assault",
    "harm_to_property_y":          "harm_to_property",
    "dispossession_y":             "dispossession",
    "religious_encroachment_y":    "religious_encroachment",
    "restriction_of_freedoms_y":   "restriction_of_freedoms",
    "coercive_actions_y":          "coercive_actions",
    "protest_y":                   "protest",
    "multi_community_incident_y":  "multi_community_incident",
}

# Columns preserved from training schema when present in input
METADATA_COLS = [
    "row_id", "id", "date", "year", "source_file",
    "area", "location_name", "location_type", "governorate",
    "jurisdiction", "perpetrator", "description",
]

# String subtype columns — preserved as-is (not predicted by model)
SUBTYPE_COLS = [
    "physical_assault", "harm_to_property", "dispossession",
    "coercive_actions", "restriction_of_freedoms",
]

# Values that map to positive (1) when entered by a human reviewer
POSITIVE_VALUES = {"1", "yes", "true", "y", "x", "✓", "1.0"}
NEGATIVE_VALUES = {"0", "no", "false", "n", "0.0"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ===========================================================================
# Input loading
# ===========================================================================

def load_sheet(path: Path) -> pd.DataFrame:
    """
    Load a reviewed sheet from Excel (.xlsx) or CSV.

    All values are kept as strings initially; normalization happens per-column.
    """
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str).fillna("")
        log.info("Loaded %d rows from Excel: %s", len(df), path)
    elif ext == ".csv":
        df = pd.read_csv(path, dtype=str).fillna("")
        log.info("Loaded %d rows from CSV: %s", len(df), path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .xlsx or .csv.")
    return df


def detect_sheet_type(df: pd.DataFrame) -> str:
    """
    Auto-detect whether the sheet is a full review sheet or compact review sheet.

    Full sheet:    has pred_raid column (individual model predictions included)
    Compact sheet: has predicted_labels column but no pred_raid
    """
    if "pred_raid" in df.columns:
        return "full"
    if "predicted_labels" in df.columns:
        return "compact"
    raise ValueError(
        "Cannot determine sheet type. Expected either 'pred_raid' (full sheet) "
        "or 'predicted_labels' (compact sheet) column."
    )


def load_predictions_audit(path: Path) -> pd.DataFrame:
    """
    Load the predictions_{ts}.csv audit file produced by predict_and_prepare_review.py.

    Used when the input is a compact review sheet (which lacks pred_* columns).
    Joins to the reviewed sheet on row_id.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    log.info("Loaded predictions audit: %d rows from %s", len(df), path)
    return df


def merge_compact_with_predictions(
    reviewed_df: pd.DataFrame, predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join compact review sheet with predictions audit file on row_id.

    Adds pred_{short} columns from predictions_df into reviewed_df so that
    the rest of the pipeline can treat it uniformly like a full review sheet.
    """
    pred_cols = ["row_id"] + [f"pred_{LABEL_SHORT[l]}" for l in LABELS]
    missing = [c for c in pred_cols if c not in predictions_df.columns]
    if missing:
        raise ValueError(
            f"predictions file is missing expected columns: {missing}\n"
            f"Make sure you are passing the predictions_{{ts}}.csv produced by "
            f"predict_and_prepare_review.py."
        )

    merged = reviewed_df.merge(
        predictions_df[pred_cols],
        on="row_id",
        how="left",
        suffixes=("", "_pred"),
    )
    n_unmatched = merged[[f"pred_{LABEL_SHORT[LABELS[0]]}"]].isna().sum().item()
    if n_unmatched:
        log.warning(
            "%d rows in reviewed sheet could not be matched to predictions "
            "(missing row_id in predictions file). These will produce model "
            "predictions of NaN — check row_id alignment.",
            n_unmatched,
        )
    return merged


# ===========================================================================
# Label normalization
# ===========================================================================

def normalize_human_label(val) -> Optional[int]:
    """
    Normalize a human reviewer's label entry to 0, 1, or None.

    None means the cell was left blank — the model prediction will be used.
    Accepts a broad set of natural inputs so RAs can use whatever feels natural.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    v = str(val).strip().lower()
    if v == "":
        return None
    if v in POSITIVE_VALUES:
        return 1
    if v in NEGATIVE_VALUES:
        return 0
    log.warning(
        "Unrecognized human label value '%s' — treating as no input (model prediction used).",
        val,
    )
    return None


def normalize_pred_label(val) -> Optional[int]:
    """Normalize a model prediction value (from pred_* column) to 0 or 1."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    v = str(val).strip()
    if v in {"1", "1.0"}:
        return 1
    if v in {"0", "0.0"}:
        return 0
    log.warning("Unexpected pred value '%s' — treating as 0.", val)
    return 0


# ===========================================================================
# Core merge logic
# ===========================================================================

def resolve_row_labels(
    row: pd.Series,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Resolve the final label for each of the 10 labels for a single row.

    Resolution rule:
        review_status == "reviewed" AND human_{label} is non-empty
            → human label wins
        otherwise
            → model prediction is used

    Returns:
        final_labels  -- {label_key: 0_or_1}
        label_sources -- {label_key: "human" or "model"}
    """
    is_reviewed = str(row.get("review_status", "")).strip().lower() == "reviewed"
    final_labels: Dict[str, int] = {}
    label_sources: Dict[str, str] = {}

    for label in LABELS:
        short    = LABEL_SHORT[label]
        human_val = normalize_human_label(row.get(f"human_{short}", ""))
        pred_val  = normalize_pred_label(row.get(f"pred_{short}", "0"))

        if is_reviewed and human_val is not None:
            final_labels[label]  = human_val
            label_sources[label] = "human"
        else:
            # Fall back to model prediction (default 0 if pred is missing)
            final_labels[label]  = pred_val if pred_val is not None else 0
            label_sources[label] = "model"

    return final_labels, label_sources


def build_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply resolve_row_labels to every row and assemble the output DataFrame.

    Output columns follow the training/train.csv schema plus audit columns.
    """
    records = []

    for _, row in df.iterrows():
        final_labels, label_sources = resolve_row_labels(row)
        human_override_count = sum(1 for s in label_sources.values() if s == "human")

        record: dict = {}

        # --- Metadata columns (pass through; fill with "" if absent) ---
        for col in METADATA_COLS:
            record[col] = str(row.get(col, "")).strip()

        # --- String subtype columns (pass through) ---
        for col in SUBTYPE_COLS:
            record[col] = str(row.get(col, "")).strip()

        # --- Final binary labels ---
        for label in LABELS:
            record[label] = final_labels[label]

        # --- Per-label source audit ---
        for label in LABELS:
            record[f"label_source_{LABEL_SHORT[label]}"] = label_sources[label]

        # --- Summary audit fields ---
        record["human_override_count"] = human_override_count
        record["review_status"]        = str(row.get("review_status", "")).strip()
        record["reviewed_by"]          = str(row.get("reviewed_by", "")).strip()
        record["reviewer_notes"]       = str(row.get("reviewer_notes", "")).strip()
        record["model_used"]           = str(row.get("model_used", "")).strip()
        record["run_timestamp"]        = str(row.get("run_timestamp", "")).strip()

        records.append(record)

    out = pd.DataFrame(records)

    # Ensure label columns are integers (not object dtype)
    for label in LABELS:
        out[label] = pd.to_numeric(out[label], errors="coerce").fillna(0).astype(int)

    return out


# ===========================================================================
# Master dataset append
# ===========================================================================

def append_to_master(new_df: pd.DataFrame, master_path: Path) -> pd.DataFrame:
    """
    Append finalized rows to the master training dataset.

    De-duplicates on row_id: if a row_id already exists in the master,
    the newer reviewed version replaces it (with a warning showing the count).

    If the master file does not exist yet, it is created from scratch.
    """
    if master_path.exists():
        master_df = pd.read_csv(master_path, dtype=str)
        log.info("Master dataset loaded: %d rows from %s", len(master_df), master_path)

        # Identify overlapping row_ids
        existing_ids = set(master_df["row_id"].values)
        new_ids      = set(new_df["row_id"].values)
        overlap      = existing_ids & new_ids

        if overlap:
            log.warning(
                "%d row_id(s) already exist in master and will be replaced "
                "with the reviewed version: %s%s",
                len(overlap),
                ", ".join(sorted(overlap)[:5]),
                " ..." if len(overlap) > 5 else "",
            )
            master_df = master_df[~master_df["row_id"].isin(overlap)]

        combined = pd.concat([master_df, new_df], ignore_index=True)
        n_new = len(new_df) - len(overlap)
        log.info(
            "Master dataset updated: %d existing + %d new + %d replaced = %d total rows",
            len(master_df), n_new, len(overlap), len(combined),
        )
    else:
        combined = new_df.copy()
        log.info(
            "Master dataset created from scratch: %d rows → %s",
            len(combined), master_path,
        )

    master_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(master_path, index=False, encoding="utf-8")
    log.info("Master dataset saved: %s", master_path)
    return combined


# ===========================================================================
# Summary printer
# ===========================================================================

def print_summary(
    final_df: pd.DataFrame,
    output_path: Path,
    master_path: Optional[Path],
) -> None:
    """Print a plain-text summary to stdout."""
    n_rows     = len(final_df)
    n_reviewed = (final_df["review_status"] == "reviewed").sum()
    n_model    = (final_df["review_status"] != "reviewed").sum()
    n_overrides = final_df["human_override_count"].astype(int).sum()

    print("\n" + "=" * 62)
    print("  MERGE SUMMARY")
    print("=" * 62)
    print(f"  Total rows processed  : {n_rows}")
    print(f"  Rows marked reviewed  : {n_reviewed}  ({100*n_reviewed/max(n_rows,1):.1f}%)")
    print(f"  Rows using model only : {n_model}  ({100*n_model/max(n_rows,1):.1f}%)")
    print(f"  Human label overrides : {n_overrides}  (across all labels)")
    print()
    print("  Final label positive rates:")
    for label in LABELS:
        short = LABEL_SHORT[label]
        rate  = final_df[label].astype(int).mean()
        human_rate = (
            (final_df[f"label_source_{short}"] == "human").mean()
        )
        print(
            f"    {short:32s} {rate*100:5.1f}%  "
            f"({human_rate*100:.0f}% from human)"
        )
    print()
    print(f"  Output: {output_path}")
    if master_path:
        print(f"  Master: {master_path}")
    print("=" * 62 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge human-reviewed labels back into a finalized training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reviewed", "-r",
        default=None,
        help=(
            "Path to the reviewed sheet (.xlsx or .csv). "
            "Accepts either the full review sheet (review_{ts}) or the "
            "compact sheet (review_compact_{ts}). "
            "Mutually exclusive with --from-sheets."
        ),
    )
    parser.add_argument(
        "--from-sheets",
        action="store_true",
        help=(
            "Pull reviewed rows directly from the Google Sheets 'Review' tab "
            "(rows where review_status == 'reviewed'). "
            "Mutually exclusive with --reviewed."
        ),
    )
    parser.add_argument(
        "--predictions", "-p",
        default=None,
        help=(
            "Path to the predictions audit file (predictions_{ts}.csv). "
            "Required when --reviewed is a compact sheet. "
            "Not needed for full review sheets."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output path for the finalized reviewed_incidents CSV. "
            "Default: data/reviewed/reviewed_incidents_{ts}.csv"
        ),
    )
    parser.add_argument(
        "--append-to", "-a",
        default=None,
        help=(
            "Path to the master training dataset CSV. "
            "If provided, finalized rows are appended (with de-duplication on row_id). "
            "Default: data/training/master_reviewed_dataset.csv"
        ),
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Skip appending to the master dataset even if --append-to is set.",
    )
    args = parser.parse_args()

    # --- Validate mutual exclusivity ---
    if args.from_sheets and args.reviewed:
        log.error("--from-sheets and --reviewed are mutually exclusive. Choose one.")
        sys.exit(1)
    if not args.from_sheets and not args.reviewed:
        log.error("Either --reviewed <path> or --from-sheets must be specified.")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data" / "reviewed" / f"reviewed_incidents_{ts}.csv"
    )
    master_path = (
        None if args.no_append
        else Path(args.append_to) if args.append_to
        else PROJECT_ROOT / "data" / "training" / "master_reviewed_dataset.csv"
    )

    # --- Load reviewed data ---
    if args.from_sheets:
        log.info("Pulling reviewed rows from Google Sheets Review tab ...")
        try:
            from data_pipeline.database.sheets_interface import get_reviewed_rows
            reviewed_df = get_reviewed_rows()
        except Exception as exc:
            log.error("Failed to read from Google Sheets: %s", exc)
            sys.exit(1)
        if reviewed_df.empty:
            log.warning("No rows with review_status='reviewed' found in Sheets Review tab.")
            sys.exit(0)
        log.info("Loaded %d reviewed rows from Sheets.", len(reviewed_df))
    else:
        reviewed_path = Path(args.reviewed)
        if not reviewed_path.exists():
            log.error("Reviewed file not found: %s", reviewed_path)
            sys.exit(1)
        reviewed_df = load_sheet(reviewed_path)

    # --- Validate required columns ---
    if "row_id" not in reviewed_df.columns:
        log.error("Reviewed sheet is missing 'row_id' column.")
        sys.exit(1)
    if "review_status" not in reviewed_df.columns:
        log.warning(
            "'review_status' column not found — all rows will use model predictions."
        )
        reviewed_df["review_status"] = ""

    # --- Auto-detect sheet type and add pred_* columns if needed ---
    sheet_type = detect_sheet_type(reviewed_df)
    log.info("Detected sheet type: %s", sheet_type)

    if sheet_type == "compact":
        if args.predictions is None:
            log.error(
                "Compact review sheet detected but --predictions was not provided.\n"
                "Pass the predictions_{ts}.csv audit file produced alongside the compact sheet.\n"
                "This file is needed so labels not overridden by an RA fall back to the "
                "model's predictions rather than defaulting to 0.\n"
                "  --predictions data/review_exports/predictions_{ts}.csv"
            )
            sys.exit(1)
        predictions_path = Path(args.predictions)
        if not predictions_path.exists():
            log.error("Predictions file not found: %s", predictions_path)
            sys.exit(1)
        predictions_df = load_predictions_audit(predictions_path)
        reviewed_df    = merge_compact_with_predictions(reviewed_df, predictions_df)

    # --- Resolve labels and build final dataset ---
    log.info("Resolving labels for %d rows ...", len(reviewed_df))
    final_df = build_final_dataset(reviewed_df)

    # --- Export finalized CSV ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Finalized dataset saved: %s  (%d rows)", output_path, len(final_df))

    # --- Append to master dataset ---
    if master_path is not None:
        append_to_master(final_df, master_path)

    # --- Summary ---
    print_summary(final_df, output_path, master_path)


if __name__ == "__main__":
    main()
