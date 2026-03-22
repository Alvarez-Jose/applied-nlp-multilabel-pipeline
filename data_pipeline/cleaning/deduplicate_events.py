"""
data_pipeline/cleaning/deduplicate_events.py

Detect and remove near-duplicate incident reports within a batch CSV.

Two reports are considered potential duplicates when they share the same date
and governorate AND have a TF-IDF cosine similarity above --threshold on their
description text. This catches cases where WAFA publishes two articles about
the same raid, or where the same incident is scraped twice from slightly
different URLs.

When a duplicate pair is found, the row with the *shorter* description is
removed (the longer one contains more information).

Usage:
    # Preview duplicates without removing (dry run):
    python data_pipeline/cleaning/deduplicate_events.py \\
        --input data/incoming/batch_20250321_120000.csv \\
        --dry-run

    # Deduplicate and write cleaned file to a new path:
    python data_pipeline/cleaning/deduplicate_events.py \\
        --input data/incoming/batch_20250321_120000.csv \\
        --output data/incoming/batch_20250321_deduped.csv

    # Deduplicate in place (overwrites input file):
    python data_pipeline/cleaning/deduplicate_events.py \\
        --input data/incoming/batch_20250321_120000.csv

    # Lower threshold for stricter deduplication (default: 0.85):
    python data_pipeline/cleaning/deduplicate_events.py \\
        --input data/incoming/batch.csv --threshold 0.75

    # Save a report of all duplicate pairs found:
    python data_pipeline/cleaning/deduplicate_events.py \\
        --input data/incoming/batch.csv \\
        --report data/incoming/duplicates_report.csv

Notes:
    - Similarity is computed only within rows that share the same date AND
      governorate, so cross-region pairs are never compared. This keeps
      runtime fast and avoids false positives.
    - A threshold of 0.85 catches near-verbatim duplicates reliably.
      Lower thresholds (0.70-0.80) catch paraphrased duplicates but increase
      false positive risk on genuinely distinct events.
"""

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_duplicates(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """
    Identify near-duplicate rows using TF-IDF cosine similarity on descriptions,
    grouped by (date, governorate).

    Returns a list of duplicate pair dicts sorted by similarity descending:
        row_id_a, row_id_b, similarity, date, governorate,
        description_a (truncated), description_b (truncated),
        idx_a, idx_b   <- original DataFrame row indices, used for removal
    """
    if "description" not in df.columns:
        log.error("No 'description' column in input — cannot deduplicate.")
        return []

    if len(df) < 2:
        return []

    texts = df["description"].fillna("").astype(str).tolist()

    # Fit a single TF-IDF vectorizer over all texts
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        log.warning("TF-IDF vectorization failed — descriptions may all be empty.")
        return []

    # Build group index: (date_str, governorate_lower) -> [row indices]
    dates = (
        df.get("date", pd.Series([""] * len(df)))
        .fillna("").astype(str).str.strip().str[:10]
    )
    govs = (
        df.get("governorate", pd.Series([""] * len(df)))
        .fillna("").astype(str).str.strip().str.lower()
    )

    groups: dict[tuple, list[int]] = {}
    for i in range(len(df)):
        key = (dates.iloc[i], govs.iloc[i])
        groups.setdefault(key, []).append(i)

    row_ids = (
        df["row_id"].astype(str).tolist()
        if "row_id" in df.columns
        else [str(i) for i in range(len(df))]
    )

    duplicates: list[dict] = []
    for (date, gov), idxs in groups.items():
        if len(idxs) < 2:
            continue
        sub_matrix = tfidf_matrix[idxs]
        sims = cosine_similarity(sub_matrix)

        for pos_a, pos_b in combinations(range(len(idxs)), 2):
            sim = float(sims[pos_a, pos_b])
            if sim < threshold:
                continue
            i, j = idxs[pos_a], idxs[pos_b]
            duplicates.append({
                "row_id_a":      row_ids[i],
                "row_id_b":      row_ids[j],
                "similarity":    round(sim, 4),
                "date":          date,
                "governorate":   gov,
                "description_a": texts[i][:200],
                "description_b": texts[j][:200],
                "idx_a":         i,
                "idx_b":         j,
            })

    return sorted(duplicates, key=lambda x: x["similarity"], reverse=True)


def deduplicate(df: pd.DataFrame, duplicates: list[dict]) -> tuple[pd.DataFrame, set[int]]:
    """
    Remove one row from each duplicate pair, keeping the longer description.
    If a row appears in multiple pairs, it is only removed once.

    Returns (cleaned_df, set_of_removed_indices).
    """
    remove: set[int] = set()

    for pair in duplicates:
        a, b = pair["idx_a"], pair["idx_b"]
        # Skip if one of the pair was already removed by an earlier pair
        if a in remove or b in remove:
            continue
        len_a = len(str(df.iloc[a].get("description", "") or ""))
        len_b = len(str(df.iloc[b].get("description", "") or ""))
        # Keep the longer description; remove the shorter one
        remove.add(b if len_a >= len_b else a)

    clean_df = df.drop(index=list(remove)).reset_index(drop=True)
    return clean_df, remove


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(df_original: pd.DataFrame, duplicates: list[dict], removed: set[int] | None = None) -> None:
    print()
    print("=" * 68)
    print("  DEDUPLICATION REPORT")
    print("=" * 68)
    print(f"  Input rows      : {len(df_original)}")
    print(f"  Duplicate pairs : {len(duplicates)}")
    if removed is not None:
        print(f"  Rows removed    : {len(removed)}")
        print(f"  Rows kept       : {len(df_original) - len(removed)}")

    if duplicates:
        print()
        print(f"  {'Sim':>6}  {'Date':<12}  {'Gov':<15}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*15}")
        for pair in duplicates[:20]:  # cap display at 20
            print(f"  {pair['similarity']:>6.3f}  {pair['date']:<12}  {pair['governorate']:<15}")
            print(f"    A [{pair['row_id_a']}]: {pair['description_a'][:90]}...")
            print(f"    B [{pair['row_id_b']}]: {pair['description_b'][:90]}...")
            print()
        if len(duplicates) > 20:
            print(f"  ... and {len(duplicates) - 20} more pairs (use --report to save full list)")
    else:
        print()
        print("  No duplicates found above the threshold.")

    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and remove near-duplicate incident reports in a batch CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input batch CSV (e.g. data/incoming/batch_{ts}.csv).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for the deduplicated CSV. Default: overwrites --input.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold (0–1). Pairs at or above this are duplicates. Default: {DEFAULT_THRESHOLD}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show duplicate pairs without writing any output files.",
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="Optional path to save the full duplicate pairs table as CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("File not found: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    log.info("Loaded %d rows from %s", len(df), input_path)

    duplicates = find_duplicates(df, threshold=args.threshold)

    if args.dry_run:
        print_report(df, duplicates, removed=None)
        log.info("--dry-run: no files written.")
        return

    clean_df, removed = deduplicate(df, duplicates)
    print_report(df, duplicates, removed=removed)

    # Write deduplicated CSV
    out_path = Path(args.output) if args.output else input_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path, index=False, encoding="utf-8")
    log.info("Wrote %d rows to %s", len(clean_df), out_path)

    # Optionally write duplicate pairs report
    if args.report and duplicates:
        report_rows = [
            {k: v for k, v in pair.items() if k not in ("idx_a", "idx_b")}
            for pair in duplicates
        ]
        pd.DataFrame(report_rows).to_csv(args.report, index=False, encoding="utf-8")
        log.info("Saved duplicate pairs report to %s", args.report)


if __name__ == "__main__":
    main()
