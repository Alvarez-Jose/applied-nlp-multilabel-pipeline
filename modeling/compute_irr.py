"""
modeling/compute_irr.py

Compute inter-rater reliability (IRR) metrics for rows marked double_review=TRUE
in a reviewed compact sheet. Uses Cohen's Kappa per label and macro-average Kappa
as the overall score.

Usage:
    # From a local reviewed CSV:
    python modeling/compute_irr.py --reviewed data/review_exports/review_compact_{ts}.csv

    # From the Google Sheets Review tab (pulls all reviewed rows):
    python modeling/compute_irr.py --from-sheets

    # Custom output path for the JSON summary:
    python modeling/compute_irr.py --reviewed review_compact.csv --output irr_2026.json

Outputs:
    - Prints a per-label IRR table to stdout
    - Saves a JSON summary to data/review_exports/irr_latest.json (used by the dashboard)

Kappa interpretation guide:
    < 0.20   Slight agreement
    0.21-0.40  Fair
    0.41-0.60  Moderate
    0.61-0.80  Substantial
    0.81-1.00  Almost perfect

Requirements:
    numpy, pandas (already in requirements.txt)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_SHORT_NAMES = [
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

LABEL_DISPLAY = {
    "raid":                    "Raid",
    "arrest_detention":        "Arrest/Detention",
    "physical_assault":        "Physical Assault",
    "harm_to_property":        "Harm to Property",
    "dispossession":           "Dispossession",
    "religious_encroachment":  "Religious Encroachment",
    "restriction_of_freedoms": "Restriction of Freedoms",
    "coercive_actions":        "Coercive Actions",
    "protest":                 "Protest",
    "multi_community_incident":"Multi-Community",
}

IRR_OUTPUT = PROJECT_ROOT / "data" / "review_exports" / "irr_latest.json"

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _parse_binary(val) -> int | None:
    """Convert a cell value (1/0/yes/no/blank) to int or None if missing."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    s = str(val).strip().lower()
    if s in {"1", "yes", "y", "true"}:
        return 1
    if s in {"0", "no", "n", "false"}:
        return 0
    return None  # blank or unrecognised


def _cohen_kappa(y1: list, y2: list) -> float | None:
    """
    Compute Cohen's Kappa for two binary rating lists.
    Returns None when undefined (e.g. both raters always agree, zero variance).
    """
    a = np.asarray(y1, dtype=float)
    b = np.asarray(y2, dtype=float)
    n = len(a)
    if n == 0:
        return None
    p_o = float(np.mean(a == b))          # observed agreement
    p1  = float(np.mean(a))               # rater 1 positive rate
    p2  = float(np.mean(b))               # rater 2 positive rate
    p_e = p1 * p2 + (1 - p1) * (1 - p2)  # expected agreement by chance
    if p_e >= 1.0:
        return None                        # degenerate: all labels identical
    return (p_o - p_e) / (1 - p_e)


def compute_irr(df: pd.DataFrame) -> dict:
    """
    Compute per-label IRR for double_review rows where both raters filled labels.

    Returns a dict:
        macro_kappa           : float | None
        total_overlap_rows    : int
        n_labels_evaluated    : int
        per_label             : {short_name: {n_pairs, pct_agreement, kappa}}
    """
    if "double_review" not in df.columns:
        log.error(
            "Column 'double_review' not found. "
            "Batches must be created with --overlap-pct > 0 to generate overlap rows."
        )
        return {}

    mask = df["double_review"].astype(str).str.strip().str.upper() == "TRUE"
    overlap = df[mask].copy()
    log.info("Rows marked double_review=TRUE: %d / %d", len(overlap), len(df))

    if overlap.empty:
        log.warning("No double_review rows found. Nothing to compute.")
        return {}

    per_label: dict = {}
    for short in LABEL_SHORT_NAMES:
        col1 = f"human_{short}"
        col2 = f"human2_{short}"

        if col1 not in overlap.columns or col2 not in overlap.columns:
            log.debug("Skipping %s — columns %s / %s not in sheet.", short, col1, col2)
            continue

        pairs = [
            (_parse_binary(row[col1]), _parse_binary(row[col2]))
            for _, row in overlap.iterrows()
        ]
        # Keep only rows where both raters filled a value
        pairs = [(v1, v2) for v1, v2 in pairs if v1 is not None and v2 is not None]

        if not pairs:
            per_label[short] = {"n_pairs": 0, "pct_agreement": None, "kappa": None}
            continue

        y1 = [p[0] for p in pairs]
        y2 = [p[1] for p in pairs]
        pct   = float(np.mean(np.array(y1) == np.array(y2)))
        kappa = _cohen_kappa(y1, y2)

        per_label[short] = {
            "n_pairs":       len(pairs),
            "pct_agreement": round(pct, 4),
            "kappa":         round(kappa, 4) if kappa is not None else None,
        }

    defined_kappas = [v["kappa"] for v in per_label.values() if v.get("kappa") is not None]
    macro_kappa    = round(float(np.mean(defined_kappas)), 4) if defined_kappas else None

    return {
        "macro_kappa":         macro_kappa,
        "total_overlap_rows":  int(len(overlap)),
        "n_labels_evaluated":  len(defined_kappas),
        "per_label":           per_label,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(summary: dict) -> None:
    """Print a readable IRR table to stdout."""
    print()
    print("=" * 66)
    print("  INTER-RATER RELIABILITY REPORT")
    print("=" * 66)

    if not summary:
        print("  No data available.")
        print("=" * 66)
        return

    macro  = summary.get("macro_kappa")
    n_rows = summary.get("total_overlap_rows", 0)
    n_eval = summary.get("n_labels_evaluated", 0)

    print(f"  Overlap rows evaluated  : {n_rows}")
    print(f"  Labels with enough data : {n_eval} / {len(LABEL_SHORT_NAMES)}")
    if macro is not None:
        print(f"  Macro Cohen's Kappa     : {macro:.3f}")
    else:
        print(f"  Macro Cohen's Kappa     : -- (insufficient data)")

    print()
    print(f"  {'Label':<30} {'N pairs':>8}  {'% Agree':>8}  {'Kappa':>8}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*8}")

    for short, stats in summary.get("per_label", {}).items():
        name  = LABEL_DISPLAY.get(short, short)
        n     = stats.get("n_pairs", 0)
        pct   = f"{stats['pct_agreement']*100:.1f}%" if stats.get("pct_agreement") is not None else "--"
        kappa = f"{stats['kappa']:.3f}"              if stats.get("kappa") is not None else "--"
        print(f"  {name:<30} {n:>8}  {pct:>8}  {kappa:>8}")

    print()
    print("  Kappa guide:  <0.20 slight  |  0.21-0.40 fair  |  0.41-0.60 moderate")
    print("                0.61-0.80 substantial  |  0.81-1.00 almost perfect")
    print("=" * 66)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute inter-rater reliability (Cohen's Kappa) for double-reviewed rows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--reviewed",
        help="Path to a reviewed compact CSV (review_compact_{ts}.csv).",
    )
    source.add_argument(
        "--from-sheets",
        action="store_true",
        help="Pull all reviewed rows directly from the Google Sheets Review tab.",
    )
    parser.add_argument(
        "--output",
        default=str(IRR_OUTPUT),
        help=f"Path to save the JSON summary. Default: {IRR_OUTPUT}",
    )
    args = parser.parse_args()

    # --- Load data ---
    if args.from_sheets:
        from data_pipeline.database.sheets_interface import get_reviewed_rows
        log.info("Loading reviewed rows from Google Sheets Review tab ...")
        df = get_reviewed_rows()
        if df.empty:
            log.warning("No reviewed rows found in Sheets.")
            sys.exit(0)
        log.info("Loaded %d reviewed rows from Sheets.", len(df))
    else:
        path = Path(args.reviewed)
        if not path.exists():
            log.error("File not found: %s", path)
            sys.exit(1)
        df = pd.read_csv(path, encoding="utf-8-sig")
        log.info("Loaded %d rows from %s", len(df), path)

    # --- Compute and display ---
    summary = compute_irr(df)
    print_table(summary)

    # --- Save JSON for dashboard ---
    if summary:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info("IRR summary saved to %s", out_path)


if __name__ == "__main__":
    main()
