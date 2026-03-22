"""
run_pipeline.py

Single-command orchestrator for the Palestine Violence Archive lab pipeline.

Stages:
    1. ingest     — Scrape new articles from WAFA → Google Sheets Reports tab
    2. export     — Export Reports tab → data/incoming/batch_{ts}.csv
    3. predict    — Run model → data/review_exports/ (review sheet + predictions)
    4. retrain    — Rebuild training data + retrain baseline model
                    (run manually after human review + merge step)

Usage:
    # Full ingest + export + predict (normal daily run):
    python run_pipeline.py

    # Skip scraping (use only what is already in Sheets):
    python run_pipeline.py --skip-ingest

    # Use DeBERTa instead of baseline:
    python run_pipeline.py --model deberta

    # Push predictions to Google Sheets Review tab after predicting:
    python run_pipeline.py --push-to-sheets

    # Re-export all reports (ignore already-exported tracking):
    python run_pipeline.py --skip-ingest --export-all

    # Preview each step without writing anything:
    python run_pipeline.py --dry-run

    # Retrain after human review is done:
    python run_pipeline.py --retrain-only

Human steps (not automated — happen between predict and retrain):
    1. Open data/review_exports/review_compact_{ts}.xlsx in Excel
    2. RAs fill human_{label} columns and set review_status = "reviewed"
    3. Save to data/reviewed/reviewed_{ts}.xlsx
    4. Merge reviewed labels (local Excel file):
           python modeling/merge_reviewed_labels.py \\
               --reviewed data/reviewed/reviewed_{ts}.xlsx \\
               --predictions data/review_exports/predictions_{ts}.csv

       Or pull directly from Google Sheets Review tab:
           python modeling/merge_reviewed_labels.py \\
               --from-sheets \\
               --predictions data/review_exports/predictions_{ts}.csv
    5. When enough reviewed data accumulates:
           python run_pipeline.py --retrain-only

Requirements:
    pip install pandas  (all other deps already installed)
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_dir = PROJECT_ROOT / "logs"
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"pipeline_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
log.info("Log file: %s", _log_file)

PYTHON = sys.executable


def run(cmd: list, step_name: str, dry_run: bool) -> bool:
    """
    Run a subprocess command. Returns True on success, False on failure.
    In dry-run mode, just prints the command without executing.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    if dry_run:
        log.info("[DRY RUN] Would run: %s", cmd_str)
        return True

    log.info("Running: %s", cmd_str)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log.error("Step '%s' failed (exit code %d).", step_name, result.returncode)
        return False
    return True


def find_latest_incoming(incoming_dir: Path) -> Path | None:
    """Return the most recently modified batch CSV in data/incoming/."""
    csvs = sorted(
        incoming_dir.glob("batch_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return csvs[0] if csvs else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Palestine Violence Archive — full lab pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip the scraping step (use what is already in Sheets).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip the Sheets → incoming CSV export step. Uses the latest existing batch file.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip the model prediction step.",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "deberta"],
        default="baseline",
        help="Model to use for predictions. Default: baseline.",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Re-export all Reports (ignore already-exported tracking).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Cap the export batch at N rows. Recommended: 100-200 to keep review manageable.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help=(
            "Run deduplication on the batch CSV before prediction. "
            "Removes near-duplicate articles (same date + governorate + high text similarity). "
            "Uses default threshold of 0.85."
        ),
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for --dedup. Default: 0.85.",
    )
    parser.add_argument(
        "--retrain-only",
        action="store_true",
        help="Only run the retraining steps (export_training_data + train_baseline). "
             "Use after human review + merge is complete.",
    )
    parser.add_argument(
        "--push-to-sheets",
        action="store_true",
        help="After predicting, push the compact review sheet to the Google Sheets 'Review' tab.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what each step would do without executing.",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    incoming_dir = PROJECT_ROOT / "data" / "incoming"

    print()
    print("=" * 62)
    print("  PALESTINE VIOLENCE ARCHIVE — PIPELINE RUN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()

    # -----------------------------------------------------------------------
    # RETRAIN-ONLY MODE
    # -----------------------------------------------------------------------
    if args.retrain_only:
        log.info("=== RETRAIN MODE ===")
        ok = run(
            [PYTHON, "modeling/training/export_training_data.py"],
            "export_training_data", args.dry_run,
        )
        if not ok:
            sys.exit(1)

        ok = run(
            [PYTHON, "modeling/training/train_baseline_multilabel.py"],
            "train_baseline", args.dry_run,
        )
        if not ok:
            log.warning("Baseline training failed — check the log above.")

        print()
        print("Retrain complete. Next:")
        print("  - Compare new metrics vs previous baseline_meta.json")
        print("  - If metrics hold, retrain DeBERTa:")
        print("    python modeling/training/train_transformer_multilabel.py")
        return

    # -----------------------------------------------------------------------
    # STEP 1 — INGEST
    # -----------------------------------------------------------------------
    if not args.skip_ingest:
        log.info("=== STEP 1: Ingest (scrape WAFA → Sheets) ===")
        ok = run(
            [PYTHON, "data_pipeline/scraping/ingest_reports.py"],
            "ingest_reports", args.dry_run,
        )
        if not ok:
            log.warning("Ingest failed — continuing with existing Sheets data.")
    else:
        log.info("=== STEP 1: Ingest skipped ===")

    # -----------------------------------------------------------------------
    # STEP 2 — EXPORT Sheets Reports → incoming CSV
    # -----------------------------------------------------------------------
    incoming_csv: Path | None = None

    if not args.skip_export:
        log.info("=== STEP 2: Export Reports → data/incoming/ ===")
        incoming_csv = incoming_dir / f"batch_{ts}.csv"
        export_cmd = [
            PYTHON, "data_pipeline/export_reports_to_incoming.py",
            "--output", str(incoming_csv),
        ]
        if args.export_all:
            export_cmd.append("--all")
        if args.max_rows is not None:
            export_cmd.extend(["--max-rows", str(args.max_rows)])
        if args.dry_run:
            export_cmd.append("--dry-run")

        ok = run(export_cmd, "export_reports_to_incoming", args.dry_run)
        if not ok:
            log.error("Export step failed. Stopping.")
            sys.exit(1)

        # If the file wasn't actually written (no new reports), fall back to latest
        if not args.dry_run and not incoming_csv.exists():
            log.info("No new reports exported. Checking for existing batch files ...")
            incoming_csv = find_latest_incoming(incoming_dir)
            if incoming_csv:
                log.info("Using existing batch: %s", incoming_csv)
            else:
                log.warning("No batch files found in data/incoming/. Nothing to predict.")
                args.skip_predict = True
    else:
        log.info("=== STEP 2: Export skipped — using latest existing batch ===")
        incoming_csv = find_latest_incoming(incoming_dir)
        if incoming_csv:
            log.info("Found: %s", incoming_csv)
        else:
            log.warning("No batch files found in data/incoming/. Skipping prediction.")
            args.skip_predict = True

    # -----------------------------------------------------------------------
    # STEP 2b — DEDUP (optional)
    # -----------------------------------------------------------------------
    if args.dedup and incoming_csv and not args.dry_run and incoming_csv.exists():
        log.info("=== STEP 2b: Deduplication (threshold=%.2f) ===", args.dedup_threshold)
        ok = run(
            [
                PYTHON, "data_pipeline/cleaning/deduplicate_events.py",
                "--input", str(incoming_csv),
                "--threshold", str(args.dedup_threshold),
            ],
            "deduplicate_events", args.dry_run,
        )
        if not ok:
            log.warning("Deduplication step failed — continuing with original batch.")
    elif args.dedup:
        log.info("=== STEP 2b: Dedup skipped (dry-run or no batch file) ===")

    # -----------------------------------------------------------------------
    # STEP 3 — PREDICT
    # -----------------------------------------------------------------------
    if not args.skip_predict and incoming_csv:
        log.info("=== STEP 3: Predict (%s) ===", args.model)
        predict_cmd = [
            PYTHON, "modeling/predict_and_prepare_review.py",
            "--input", str(incoming_csv),
            "--model", args.model,
        ]
        if args.push_to_sheets:
            predict_cmd.append("--push-to-sheets")
        ok = run(predict_cmd, "predict_and_prepare_review", args.dry_run)
        if not ok:
            log.error("Prediction step failed.")
            sys.exit(1)
    else:
        log.info("=== STEP 3: Predict skipped ===")

    # -----------------------------------------------------------------------
    # DONE — print next steps for the RA
    # -----------------------------------------------------------------------
    print()
    print("=" * 62)
    print("  PIPELINE COMPLETE — NEXT HUMAN STEPS")
    print("=" * 62)
    print()
    print("  1. Open the review sheet:")
    print("     data/review_exports/review_compact_{timestamp}.xlsx")
    print()
    print("  2. RAs fill in:")
    print("     - human_{label} columns (1/0 or yes/no)")
    print("     - reviewer_notes (optional)")
    print("     - review_status = 'reviewed'")
    print()
    print("  3. Save the reviewed file to:")
    print("     data/reviewed/reviewed_{timestamp}.xlsx")
    print()
    print("  4. Merge reviewed labels back into training data:")
    print("     python modeling/merge_reviewed_labels.py \\")
    print("         --reviewed data/reviewed/reviewed_{timestamp}.xlsx")
    print()
    print("  5. After enough reviewed batches, retrain:")
    print("     python run_pipeline.py --retrain-only")
    print()


if __name__ == "__main__":
    main()
