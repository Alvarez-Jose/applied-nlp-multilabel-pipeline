"""
data_pipeline/export_for_analysis.py

Export the cleaned incident database to formats ready for statistical analysis:

    analysis/Palestine_Violence_Archive.sav   -- SPSS format with variable and
                                                 value labels (open in SPSS or JASP)
    analysis/Palestine_Violence_Archive.csv   -- Clean R-ready CSV (UTF-8)
    analysis/codebook_variables.csv           -- Variable dictionary for the professor

Usage:
    python data_pipeline/export_for_analysis.py

    # Export a specific year only:
    python data_pipeline/export_for_analysis.py --years 2023 2024

    # Custom output directory:
    python data_pipeline/export_for_analysis.py --output-dir analysis/export_march2026/

Requirements:
    pip install pyreadstat  (already in requirements.txt)

Notes:
    - Reads from data/processed/oppression_{year}.csv (run normalize_raw.py first
      if those files don't exist yet)
    - Binary label columns (raid, arrest_detention, etc.) are exported with
      value labels 0=No, 1=Yes for SPSS
    - Categorical columns (governorate, perpetrator, etc.) include full value labels
    - Missing values are coded as SPSS system-missing (blank in output)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis"
YEARS = [2023, 2024, 2025]

# ---------------------------------------------------------------------------
# Variable metadata — labels shown in SPSS variable view and R codebook
# ---------------------------------------------------------------------------

VARIABLE_LABELS = {
    "row_id":                        "Unique row identifier",
    "id":                            "Original incident ID from database",
    "year":                          "Year of incident",
    "date":                          "Date of incident (YYYY-MM-DD)",
    "area":                          "Geographic area (West Bank or Gaza)",
    "governorate":                   "Palestinian governorate",
    "location_name":                 "Specific location name",
    "location_type":                 "Type of location (village, city, camp, etc.)",
    "jurisdiction":                  "Oslo Accord jurisdiction (Area A, B, or C)",
    "perpetrator":                   "Primary perpetrator category",
    "palestinian_involvement":       "Type of Palestinian involvement",
    "raid":                          "Incident involved a raid by Israeli forces",
    "arrest_detention":              "Incident involved arrest or detention",
    "number_arrested":               "Number of individuals arrested",
    "physical_assault":              "Incident involved physical assault or violence",
    "killed":                        "Number of Palestinians killed",
    "number_injured":                "Number of Palestinians injured",
    "coercive_actions":              "Incident involved coercive or intimidation tactics",
    "restriction_of_freedoms":       "Incident involved restriction of movement or freedoms",
    "religious_encroachment":        "Incident involved encroachment on religious sites",
    "harm_to_property":              "Incident involved damage or destruction of property",
    "type_of_property_harmed":       "Type of property damaged (vandalism or destruction)",
    "dispossession":                 "Incident involved dispossession of land or assets",
    "type_of_property_dispossessed": "Type of property or assets dispossessed",
    "multi_community_incident":      "Incident affected multiple communities simultaneously",
    "protest":                       "Incident involved a Palestinian protest or demonstration",
    "type_of_protest":               "Type of protest action",
    "description":                   "Narrative description of the incident",
    "source_1":                      "Primary source URL or citation",
    "source_2":                      "Secondary source URL or citation",
    "source_3":                      "Tertiary source URL or citation",
}

# Value labels for binary (0/1) columns
BINARY_VALUE_LABELS = {0: "No", 1: "Yes"}

BINARY_COLS = [
    "raid", "arrest_detention", "physical_assault", "coercive_actions",
    "restriction_of_freedoms", "religious_encroachment", "harm_to_property",
    "dispossession", "multi_community_incident", "protest",
]

# Value labels for categorical columns
CATEGORICAL_VALUE_LABELS = {
    "governorate": {
        "Nablus":        "Nablus",
        "Hebron":        "Hebron",
        "Jenin":         "Jenin",
        "Ramallah":      "Ramallah and Al-Bireh",
        "Bethlehem":     "Bethlehem",
        "Tulkarm":       "Tulkarm",
        "Qalqilya":      "Qalqilya",
        "Salfit":        "Salfit",
        "Tubas":         "Tubas",
        "Jericho":       "Jericho and Al-Aghwar",
        "Jerusalem":     "Jerusalem",
        "Jordan Valley": "Jordan Valley",
    },
    "perpetrator": {
        "Israeli Soldiers":       "Israeli Military / IDF",
        "Israeli Settlers":       "Israeli Settlers",
        "Israeli Police":         "Israeli Police / Border Police",
        "Palestinian Authority":  "Palestinian Authority",
        "Israeli Citizens":       "Israeli Civilians",
    },
    "jurisdiction": {
        "Area A":  "Area A (Palestinian Authority control)",
        "Area B":  "Area B (Shared control)",
        "Area C":  "Area C (Israeli Civil and Military control)",
        "Mixed":   "Mixed jurisdiction",
        "Unknown": "Unknown",
    },
    "area": {
        "West Bank": "West Bank",
        "Gaza":      "Gaza Strip",
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_years(years: list[int]) -> pd.DataFrame:
    """Load and concatenate normalized CSVs for the requested years."""
    frames = []
    for year in years:
        path = PROCESSED_DIR / f"oppression_{year}.csv"
        if not path.exists():
            log.warning(
                "Processed file not found for %d: %s — run normalize_raw.py first. Skipping.",
                year, path,
            )
            continue
        df = pd.read_csv(path, low_memory=False)
        df["year"] = year
        frames.append(df)
        log.info("Loaded %d rows from %s", len(df), path.name)

    if not frames:
        log.error(
            "No processed files found. Run: python data_pipeline/normalize_raw.py --all"
        )
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined dataset: %d rows across %d year(s)", len(combined), len(frames))
    return combined


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the combined DataFrame for export."""
    # Ensure year column is present and positioned early
    cols = list(df.columns)
    if "year" in cols:
        cols.remove("year")
    # Insert year after row_id
    insert_after = "row_id" if "row_id" in cols else 0
    if isinstance(insert_after, str):
        pos = cols.index(insert_after) + 1
        cols.insert(pos, "year")
    df = df[cols]

    # Coerce binary columns to numeric, preserving NA
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce numeric count columns
    for col in ["number_arrested", "killed", "number_injured"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardise date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": None, "": None})

    # Keep only columns that have a variable label defined (drop pipeline internals)
    export_cols = [c for c in df.columns if c in VARIABLE_LABELS]
    # Always keep description and sources even if missing from label dict
    for extra in ["description", "source_1", "source_2", "source_3"]:
        if extra in df.columns and extra not in export_cols:
            export_cols.append(extra)

    return df[export_cols].copy()


# ---------------------------------------------------------------------------
# SPSS export
# ---------------------------------------------------------------------------

def export_spss(df: pd.DataFrame, output_path: Path) -> None:
    try:
        import pyreadstat
    except ImportError:
        log.error("pyreadstat not installed. Run: pip install pyreadstat")
        return

    # Build variable labels dict (only for columns present)
    var_labels = {col: VARIABLE_LABELS[col] for col in df.columns if col in VARIABLE_LABELS}

    # Build value labels dict
    val_labels: dict = {}
    for col in BINARY_COLS:
        if col in df.columns:
            val_labels[col] = BINARY_VALUE_LABELS
    for col, labels in CATEGORICAL_VALUE_LABELS.items():
        if col in df.columns:
            val_labels[col] = labels

    # pyreadstat needs float NaN for missing, not pd.NA
    df_spss = df.copy()
    for col in df_spss.columns:
        try:
            df_spss[col] = pd.to_numeric(df_spss[col], errors="ignore")
        except Exception:
            pass
    # Replace pd.NA with np.nan throughout
    df_spss = df_spss.fillna(value=np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pyreadstat.write_sav(
        df_spss,
        str(output_path),
        variable_value_labels=val_labels,
        column_labels=list(var_labels.values()),
    )
    log.info("SPSS file written: %s  (%d rows, %d variables)", output_path, len(df_spss), len(df_spss.columns))


# ---------------------------------------------------------------------------
# R-ready CSV export
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("R-ready CSV written: %s  (%d rows)", output_path, len(df))


# ---------------------------------------------------------------------------
# Variable codebook export
# ---------------------------------------------------------------------------

def export_codebook(df: pd.DataFrame, output_path: Path) -> None:
    """Write a variable dictionary CSV for the professor to reference."""
    rows = []
    for col in df.columns:
        n_valid   = int(df[col].notna().sum())
        n_missing = int(df[col].isna().sum())
        pct_miss  = round(n_missing / len(df) * 100, 1) if len(df) > 0 else 0

        if col in BINARY_COLS:
            values = "0 = No, 1 = Yes"
            col_type = "Binary (0/1)"
        elif col in CATEGORICAL_VALUE_LABELS:
            values = "; ".join(f"{k}" for k in CATEGORICAL_VALUE_LABELS[col].keys())
            col_type = "Categorical"
        elif col in ("number_arrested", "killed", "number_injured"):
            col_type = "Count (numeric)"
            values = f"Range: {df[col].min()} – {df[col].max()}"
        elif col == "date":
            col_type = "Date (YYYY-MM-DD)"
            values = f"{df[col].min()} to {df[col].max()}"
        elif col == "year":
            col_type = "Numeric"
            values = ", ".join(str(int(y)) for y in sorted(df[col].dropna().unique()))
        else:
            col_type = "Text"
            values = ""

        rows.append({
            "Variable":        col,
            "Label":           VARIABLE_LABELS.get(col, ""),
            "Type":            col_type,
            "Valid N":         n_valid,
            "Missing N":       n_missing,
            "Missing %":       pct_miss,
            "Values / Range":  values,
        })

    codebook_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codebook_df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Variable codebook written: %s", output_path)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 64)
    print("  EXPORT SUMMARY")
    print("=" * 64)
    print(f"  Total incidents exported : {len(df)}")

    if "year" in df.columns:
        print()
        print("  By year:")
        for year, count in df["year"].value_counts().sort_index().items():
            print(f"    {int(year)}: {count} incidents")

    if "governorate" in df.columns:
        print()
        print("  Top 5 governorates:")
        for gov, count in df["governorate"].value_counts().head(5).items():
            pct = count / len(df) * 100
            print(f"    {gov:<20} {count:>5}  ({pct:.1f}%)")

    print()
    print("  Label prevalence:")
    for col in BINARY_COLS:
        if col in df.columns:
            n = df[col].sum()
            if pd.notna(n):
                pct = n / len(df) * 100
                label = VARIABLE_LABELS.get(col, col)
                print(f"    {col:<30} {int(n):>5}  ({pct:.1f}%)")

    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export incident database to SPSS (.sav) and R-ready CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=YEARS,
        help=f"Years to include. Default: {YEARS}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--no-spss",
        action="store_true",
        help="Skip SPSS export (useful if pyreadstat is not installed).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df_raw  = load_years(args.years)
    df_clean = prepare(df_raw)

    year_str = "_".join(str(y) for y in sorted(args.years))

    # Export SPSS
    if not args.no_spss:
        export_spss(df_clean, output_dir / f"Palestine_Violence_Archive_{year_str}.sav")

    # Export R-ready CSV
    export_csv(df_clean, output_dir / f"Palestine_Violence_Archive_{year_str}.csv")

    # Export variable codebook
    export_codebook(df_clean, output_dir / "codebook_variables.csv")

    # Print summary
    print_summary(df_clean)

    print("  Files written to:", output_dir)
    print()
    print("  To open in SPSS:")
    print(f"    File > Open > Data > Palestine_Violence_Archive_{year_str}.sav")
    print()
    print("  To open in R:")
    print(f"    df <- read.csv('Palestine_Violence_Archive_{year_str}.csv')")
    print(f"    # or with Haven for labelled data:")
    print(f"    df <- haven::read_sav('Palestine_Violence_Archive_{year_str}.sav')")
    print()


if __name__ == "__main__":
    main()
