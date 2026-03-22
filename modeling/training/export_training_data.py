import re
from pathlib import Path
from typing import List

import pandas as pd


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "modeling" / "training" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_FILES = [
    ("oppression_2023.csv", 2023),
    ("oppression_2024.csv", 2024),
    ("oppression_2025.csv", 2025),
]

# Sheets export (produced by data_pipeline/export_sheets_to_csv.py).
# Included automatically when the file exists; year is derived per-row from the date column.
SHEETS_EXPORT_FILE = PROCESSED_DIR / "incidents_from_sheets.csv"


# ---------- Target normalization ----------
def norm_bool01(x) -> int:
    """For already-normalized 0/1-ish fields."""
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n", ""}:
        return 0
    # if it's some weird token, treat as present
    return 1


def norm_presence(x) -> int:
    """
    For fields where presence is represented by a non-empty string
    (e.g., physical_assault="Shooting", harm_to_property="Vandalism").
    """
    if pd.isna(x):
        return 0
    s = str(x).strip()
    if s == "" or s.upper() == "FALSE" or s == "0":
        return 0
    return 1


# ---------- Build dataset ----------
def build() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for fname, year in PROCESSED_FILES:
        path = PROCESSED_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed CSV: {path}\n"
                f"Run: python data_pipeline/normalize_raw.py --all"
            )

        df = pd.read_csv(path)

        # Add year + source
        df["year"] = year
        df["source_file"] = fname

        # Ensure expected cols exist
        required = [
            "row_id", "id", "date", "description",
            "area", "location_name", "location_type",
            "governorate", "jurisdiction", "perpetrator",
        ]
        for c in required:
            if c not in df.columns:
                df[c] = ""

        # Some optional cols used for presence labels
        for c in ["physical_assault", "coercive_actions", "restriction_of_freedoms", "harm_to_property", "dispossession"]:
            if c not in df.columns:
                df[c] = ""

        # Create label columns (names match your baseline script)
        # 0/1-style (these are already normalized in your processed files)
        for col in ["raid", "arrest_detention", "religious_encroachment", "protest", "multi_community_incident"]:
            if col not in df.columns:
                df[col] = 0
            df[col + "_y"] = df[col].apply(norm_bool01)

        # presence-style (string-valued)
        for col in ["physical_assault", "coercive_actions", "restriction_of_freedoms", "harm_to_property", "dispossession"]:
            df[col + "_y"] = df[col].apply(norm_presence)

        frames.append(df)

    # Optionally include converted oppression CSVs
    # (produced by data_pipeline/convert_oppression_db_to_incidents.py)
    for converted_path in sorted(PROCESSED_DIR.glob("oppression_*_converted.csv")):
        df_conv = pd.read_csv(converted_path, dtype=str)
        df_conv["year"] = (
            pd.to_datetime(df_conv.get("date", pd.Series(dtype=str)), errors="coerce").dt.year
        )
        n_before = len(df_conv)
        df_conv = df_conv.dropna(subset=["year"])
        df_conv["year"] = df_conv["year"].astype(int)
        if n_before > len(df_conv):
            print(f"[WARN] {converted_path.name}: dropped {n_before - len(df_conv)} rows with unparseable dates")
        df_conv["source_file"] = converted_path.name
        frames.append(df_conv)
        print(f"[OK] Included {converted_path.name}: {len(df_conv)} rows")

    # Optionally include rows exported from Google Sheets
    if SHEETS_EXPORT_FILE.exists():
        df_sheets = pd.read_csv(SHEETS_EXPORT_FILE)
        # Derive year from the date column; drop rows where year cannot be parsed
        df_sheets["year"] = (
            pd.to_datetime(df_sheets["date"], errors="coerce").dt.year
        )
        n_before = len(df_sheets)
        df_sheets = df_sheets.dropna(subset=["year"])
        df_sheets["year"] = df_sheets["year"].astype(int)
        if n_before > len(df_sheets):
            print(f"[WARN] incidents_from_sheets.csv: dropped {n_before - len(df_sheets)} rows with unparseable dates")
        df_sheets["source_file"] = "incidents_from_sheets.csv"
        frames.append(df_sheets)
        print(f"[OK] Included incidents_from_sheets.csv: {len(df_sheets)} rows")
    else:
        print("[INFO] incidents_from_sheets.csv not found — skipping. "
              "Run: python data_pipeline/export_sheets_to_csv.py")

    full = pd.concat(frames, ignore_index=True, sort=False)

    # Standardize date to string
    full["date"] = full["date"].astype(str).str.strip()

    keep_cols = [
        # ids/audit
        "row_id", "id", "date", "year", "source_file",

        # metadata
        "area", "location_name", "location_type", "governorate", "jurisdiction",
        "perpetrator",

        # text
        "description",

        # raw subtype-ish fields (optional, keep for later)
        "physical_assault", "harm_to_property", "dispossession",
        "coercive_actions", "restriction_of_freedoms",

        # labels
        "raid_y", "arrest_detention_y", "physical_assault_y",
        "harm_to_property_y", "dispossession_y", "religious_encroachment_y",
        "restriction_of_freedoms_y", "coercive_actions_y",
        "protest_y", "multi_community_incident_y",
    ]
    keep_cols = [c for c in keep_cols if c in full.columns]
    return full[keep_cols].copy()


def main():
    df = build()

    # Save canonical dataset
    out_parquet = OUT_DIR / "incidents_2023_2025.parquet"
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception as e:
        print(f"[WARN] Failed to write parquet ({e}). Install pyarrow: pip install pyarrow")

    # Time-based split: train=2023–2024, dev=2025
    train = df[df["year"].isin([2023, 2024])].copy()
    dev = df[df["year"] == 2025].copy()

    train_path = OUT_DIR / "train.csv"
    dev_path = OUT_DIR / "dev.csv"
    train.to_csv(train_path, index=False)
    dev.to_csv(dev_path, index=False)

    # Quick label prevalence report
    label_cols = [c for c in df.columns if c.endswith("_y")]
    print("\n=== DATASET BUILT ===")
    print("Processed dir:", PROCESSED_DIR)
    print("Out dir:", OUT_DIR)
    print("Rows:", len(df))
    print("Train rows:", len(train), "Dev rows:", len(dev))
    print("\nLabel prevalence (mean):")
    for c in label_cols:
        print(f"  {c}: {df[c].mean():.3f}")

    print("\nWrote:")
    print("  ", train_path)
    print("  ", dev_path)
    if out_parquet.exists():
        print("  ", out_parquet)


if __name__ == "__main__":
    main()
