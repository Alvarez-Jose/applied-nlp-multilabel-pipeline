"""
data_pipeline/export_reports_to_incoming.py

Exports scraped reports from the Google Sheets "Reports" tab into a CSV
that predict_and_prepare_review.py can run the model on.

Position in the workflow:
    ingest_reports.py              (scraper → Sheets)
    export_reports_to_incoming.py  (Sheets Reports → data/incoming/)
    predict_and_prepare_review.py  (model predictions + review sheet)
    [RA reviews in Excel]
    merge_reviewed_labels.py       (reviewed labels → training data)

Column mapping (Reports sheet → model input):
    report_id      → row_id, id
    published_date → date
    body_text      → description
    location_raw   → location_name  (structured best-guess)
    location_raw   → governorate    (extracted from known governorate list)
    actors_raw     → perpetrator    (extracted from known perpetrator list)
    url            → source_1
    source_name    → source_2

Filtering:
    - Skips rows with empty body_text
    - Skips rows already exported (tracked in data/incoming/.exported_ids.json)
    - Skips non-English articles (language != 'en')

Usage:
    # Export new reports not yet sent to model:
    python data_pipeline/export_reports_to_incoming.py

    # Re-export everything (ignore already-exported tracking):
    python data_pipeline/export_reports_to_incoming.py --all

    # Preview without writing:
    python data_pipeline/export_reports_to_incoming.py --dry-run

    # Custom output path:
    python data_pipeline/export_reports_to_incoming.py --output data/incoming/my_batch.csv
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.database.sheets_interface import get_sheet, REPORTS_SHEET_NAME  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

INCOMING_DIR  = PROJECT_ROOT / "data" / "incoming"
EXPORTED_IDS_FILE = INCOMING_DIR / ".exported_ids.json"

# ---------------------------------------------------------------------------
# Known governorate names for extraction from location_raw
# ---------------------------------------------------------------------------
GOVERNORATE_NAMES = [
    "Nablus", "Qalqilya", "Tubas", "Salfit", "Tulkarm", "Jenin",
    "Jericho", "Jordan Valley", "Ramallah", "Bethlehem", "Hebron", "Jerusalem",
]

# ---------------------------------------------------------------------------
# Known perpetrator tokens for extraction from actors_raw
# ---------------------------------------------------------------------------
PERPETRATOR_PATTERNS = [
    ("Israeli Settlers",  ["settler", "settlers"]),
    ("Israeli Police",    ["police", "border police"]),
    ("Israeli Soldiers",  ["soldier", "soldiers", "forces", "military", "idf", "army"]),
    ("Israeli Citizens",  ["citizens", "civilians"]),
    ("Palestinian Authority", ["palestinian authority", "pa ", " pa,"]),
]


def extract_governorate(location_raw: str) -> str:
    """Return the first known governorate name found in location_raw."""
    if not location_raw:
        return ""
    loc = location_raw.lower()
    for g in GOVERNORATE_NAMES:
        if g.lower() in loc:
            return g
    return ""


def extract_perpetrator(actors_raw: str) -> str:
    """Return the most specific perpetrator label found in actors_raw."""
    if not actors_raw:
        return ""
    text = actors_raw.lower()
    for label, tokens in PERPETRATOR_PATTERNS:
        if any(tok in text for tok in tokens):
            return label
    return ""


# ---------------------------------------------------------------------------
# Exported IDs tracking
# ---------------------------------------------------------------------------

def load_exported_ids() -> set:
    if EXPORTED_IDS_FILE.exists():
        with open(EXPORTED_IDS_FILE) as f:
            return set(json.load(f))
    return set()


def save_exported_ids(ids: set) -> None:
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_exported_ids()
    combined = existing | ids
    with open(EXPORTED_IDS_FILE, "w") as f:
        json.dump(sorted(combined), f, indent=2)


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(
    output_path: Path,
    export_all: bool,
    dry_run: bool,
    max_rows: int | None = None,
) -> None:

    # --- 1. Pull Reports from Sheets ---
    log.info("Reading Reports tab from Google Sheets ...")
    try:
        ws = get_sheet(REPORTS_SHEET_NAME)
        all_values = ws.get_all_values()
    except Exception as e:
        log.error("Failed to read Reports tab: %s", e)
        sys.exit(1)

    if len(all_values) < 2:
        log.warning("Reports tab is empty.")
        return

    headers = all_values[0]
    df = pd.DataFrame(all_values[1:], columns=headers).replace("", None)
    log.info("Pulled %d rows from Reports tab.", len(df))

    # --- 2. Filter ---
    already_exported = load_exported_ids()

    skipped_empty    = 0
    skipped_language = 0
    skipped_exported = 0
    rows = []

    for _, row in df.iterrows():
        report_id = str(row.get("report_id") or "").strip()
        body_text = str(row.get("body_text") or "").strip()
        language  = str(row.get("language") or "en").strip().lower()

        if not body_text or len(body_text) < 30:
            skipped_empty += 1
            continue

        if language not in {"en", "english", ""}:
            skipped_language += 1
            continue

        if not export_all and report_id in already_exported:
            skipped_exported += 1
            continue

        # --- Column mapping ---
        location_raw = str(row.get("location_raw") or "").strip()
        actors_raw   = str(row.get("actors_raw") or "").strip()

        rows.append({
            "row_id":       report_id,
            "id":           report_id,
            "date":         str(row.get("published_date") or "").strip(),
            "description":  body_text,
            "location_name": location_raw,
            "governorate":   extract_governorate(location_raw),
            "perpetrator":   extract_perpetrator(actors_raw),
            "area":          "West Bank",
            "location_type": "",
            "jurisdiction":  "",
            "source_1":      str(row.get("url") or "").strip(),
            "source_2":      str(row.get("source_name") or "").strip(),
            "title":         str(row.get("title") or "").strip(),
        })

    if not rows:
        log.info(
            "No new reports to export. "
            "(empty=%d, non-english=%d, already-exported=%d)",
            skipped_empty, skipped_language, skipped_exported,
        )
        return

    out_df = pd.DataFrame(rows)

    # Apply batch size cap — keep the most recent rows (last scraped = highest report_id)
    skipped_capped = 0
    if max_rows is not None and len(out_df) > max_rows:
        skipped_capped = len(out_df) - max_rows
        out_df = out_df.tail(max_rows).reset_index(drop=True)
        log.info("--max-rows %d applied: exporting %d rows, deferring %d.", max_rows, len(out_df), skipped_capped)

    # --- 3. Summary ---
    print("\n" + "=" * 58)
    print("  REPORTS EXPORT SUMMARY")
    print("=" * 58)
    print(f"  Total rows in Sheets   : {len(df)}")
    print(f"  Skipped (no text)      : {skipped_empty}")
    print(f"  Skipped (non-English)  : {skipped_language}")
    print(f"  Skipped (already done) : {skipped_exported}")
    print(f"  Ready to export        : {len(rows)}")
    if skipped_capped:
        print(f"  Capped by --max-rows   : {skipped_capped} deferred to next batch")
    print()
    print("  Sample descriptions (first 3):")
    for _, r in out_df.head(3).iterrows():
        preview = r["description"][:120].replace("\n", " ")
        print(f"    [{r['id']}] {preview}...")
    print()
    print(f"  Output: {output_path}")
    print("=" * 58 + "\n")

    if dry_run:
        log.info("--dry-run: no file written.")
        return

    # --- 4. Write ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Wrote %d rows to %s", len(out_df), output_path)

    # --- 5. Track exported IDs ---
    save_exported_ids({r["id"] for r in rows})
    log.info("Updated exported ID tracker: %s", EXPORTED_IDS_FILE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Export scraped Reports from Google Sheets to a model-ready CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        default=str(INCOMING_DIR / f"batch_{ts}.csv"),
        help="Output CSV path. Default: data/incoming/batch_{timestamp}.csv",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all reports, ignoring the already-exported tracker.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without writing any files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of rows to export in one batch. "
            "Keeps the most recently scraped N reports; the rest are deferred to the next run. "
            "Useful for keeping review batches manageable (recommended: 100-200)."
        ),
    )
    args = parser.parse_args()

    export(
        output_path=Path(args.output),
        export_all=args.all,
        dry_run=args.dry_run,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
