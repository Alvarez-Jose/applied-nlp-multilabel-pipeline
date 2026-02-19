from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


RAW_FILENAME_BY_YEAR = {
    2023: "Oppression database - 2023.csv",
    2024: "Oppression database - 2024.csv",
    2025: "Oppression database - 2025.csv",
}


# 1) Fix common header typos/variants BEFORE renaming to canonical
HEADER_FIXUPS = {
    # trailing spaces seen in raw exports
    "Jurisdiction ": "Jurisdiction",
    "Perpetrator ": "Perpetrator",
    "Palestinian involvement ": "Palestinian involvement",
    "Number of Injuried ": "Number of Injured",
    "Number of Injuried": "Number of Injured",

    # typo seen in 2024 diff
    "Type of Propery Disposessed": "Type of Property Dispossessed",

    # normalize hyphenated names pre-rename
    "Multi-community Incident": "Multi-Community Incident",
}


# 2) Canonical schema mapping (raw -> canonical)
# Keep canonical names stable for modeling.
COLUMN_RENAMES = {
    # IDs / metadata
    "ID": "id",
    "Date of incident": "date",
    "Location (WB or Gaza)": "area",
    "Region": "area",  # 2023 naming

    "Location name": "location_name",
    "Location": "location_name",  # 2023 naming

    "Location Type": "location_type",
    "Location type": "location_type",  # 2023 naming

    "Governorate": "governorate",
    "Jurisdiction": "jurisdiction",
    "Perpetrator": "perpetrator",
    "Palestinian involvement": "palestinian_involvement",

    # labels / structured fields
    "Raid": "raid",
    "Arrest/Detention": "arrest_detention",
    "Number Arrested": "number_arrested",
    "Physical Assault": "physical_assault",
    "Killed": "killed",
    "Number of Injured": "number_injured",
    "Coercive Actions": "coercive_actions",
    "Restriction of Freedoms": "restriction_of_freedoms",
    "Restriction of Freedoms (mechanisms)": "restriction_of_freedoms_mechanisms",
    "Religious Encroachment": "religious_encroachment",
    "Harm to Property": "harm_to_property",
    "Type of Property Harmed": "type_of_property_harmed",

    # dispossession variants
    "Disposession": "dispossession",
    "Dispossession": "dispossession",

    # property dispossessed variants/typos
    "Type of Property Dispossessed": "type_of_property_dispossessed",
    "Type of Propery Disposessed": "type_of_property_dispossessed",

    # text + extra
    "Description of Violence": "description",

    # multi-community variants
    "Multi-Community Incident": "multi_community_incident",
    "Multi-community Incident": "multi_community_incident",

    "Protest": "protest",

    # protest type variants
    "Type of Protest": "type_of_protest",
    "Type of protest": "type_of_protest",

    # sources (including 4-6 for provenance)
    "Source 1": "source_1",
    "Source 2": "source_2",
    "Source 3": "source_3",
    "Source 4": "source_4",
    "Source 5": "source_5",
    "Source 6": "source_6",
}


# 3) Enforce same columns across years
CANONICAL_OUTPUT_COLS = [
    # Unique row identifier (handles duplicate IDs)
    "row_id",
    
    # IDs / metadata
    "id", "date", "area", "location_name", "location_type", "governorate",
    "jurisdiction", "perpetrator", "palestinian_involvement",

    # labels / structured fields
    "raid", "arrest_detention", "number_arrested", "physical_assault",
    "killed", "number_injured", "coercive_actions", "restriction_of_freedoms",
    "restriction_of_freedoms_mechanisms", "religious_encroachment",
    "harm_to_property", "type_of_property_harmed", "dispossession",
    "type_of_property_dispossessed",

    # text + extra
    "description", "multi_community_incident", "protest", "type_of_protest",

    # sources (now including 4-6)
    "source_1", "source_2", "source_3", "source_4", "source_5", "source_6",
]


BOOL_MAP = {
    "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1, 1: 1, True: 1,
    "no": 0, "n": 0, "false": 0, "f": 0, "0": 0, 0: 0, False: 0,
    "": pd.NA, None: pd.NA,
}


def clean_header(col: str) -> str:
    col = str(col).strip()
    col = re.sub(r"\s+", " ", col)
    return col


def looks_booleanish(series: pd.Series) -> bool:
    if series.dtype == "bool":
        return True
    vals = series.dropna()
    if vals.empty:
        return False
    uniq = pd.unique(vals.astype(str).str.strip().str.lower())
    allowed = {"yes", "no", "true", "false", "0", "1", "y", "n", "t", "f"}
    return len(uniq) <= 6 and set(uniq).issubset(allowed)


def normalize_booleans(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        if not looks_booleanish(s):
            continue

        def _map_val(v):
            if pd.isna(v):
                return pd.NA
            if isinstance(v, str):
                return BOOL_MAP.get(v.strip().lower(), v)
            return BOOL_MAP.get(v, v)

        mapped = s.map(_map_val)
        uniq = set(pd.unique(mapped.dropna()))
        if uniq.issubset({0, 1}):
            df[col] = mapped.astype("Int64")
    return df


def normalize_one_year(year: int) -> Path:
    raw_path = RAW_DIR / RAW_FILENAME_BY_YEAR[year]
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"oppression_{year}.csv"

    df = pd.read_csv(raw_path)

    # 1) Clean headers
    df.columns = [clean_header(c) for c in df.columns]

    # 1.5) Drop junk unnamed columns (from trailing commas in CSV exports)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # 2) Fix known header variants (spaces, typos)
    df = df.rename(columns=HEADER_FIXUPS)

    # 3) Rename into canonical schema
    df = df.rename(columns=COLUMN_RENAMES)

    # 4) Normalize boolean-ish cols to 0/1
    df = normalize_booleans(df)

    # 5) Drop exact duplicate rows (safe deduplication)
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        print(f"[WARN] {year}: dropped {before-after} exact duplicate rows")

    # 6) Create unique row_id to handle duplicate IDs
    if "id" in df.columns:
        # Normalize ID for grouping (prevents NaN group weirdness)
        id_key = df["id"].fillna("MISSING_ID").astype(str)

        # How many IDs have >1 row?
        multi_ids = int((id_key.value_counts() > 1).sum())
        if multi_ids:
            print(f"[WARN] {year}: {multi_ids} IDs have multiple rows (creating row_id)")

        # Cumcount is guaranteed int; build row_id without ".0"
        suffix = df.groupby(id_key).cumcount().astype("int64").astype(str)
        df["row_id"] = (id_key + "-" + suffix).astype(str)

        # Keep cleaned id too (optional, but consistent)
        df["id"] = id_key.replace("MISSING_ID", pd.NA)
    else:
        df["row_id"] = df.index.astype(str)

    # Safety: row_id must be unique
    if df["row_id"].duplicated().any():
        raise ValueError(f"{year}: row_id is not unique — check ID/cumcount logic")

    # 7) Enforce canonical output columns (add missing as NA, keep only canonical cols)
    if CANONICAL_OUTPUT_COLS:
        for c in CANONICAL_OUTPUT_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[CANONICAL_OUTPUT_COLS]

    df.to_csv(out_path, index=False)

    print(f"[OK] Wrote: {out_path}")
    print(f"     Rows: {len(df):,}  Cols: {len(df.columns):,}")
    return out_path



def diff_columns() -> None:
    """Print column diffs between years after header fixups + canonical rename."""
    by_year = {}
    for year, fname in RAW_FILENAME_BY_YEAR.items():
        raw_path = RAW_DIR / fname
        df = pd.read_csv(raw_path, nrows=1)

        df.columns = [clean_header(c) for c in df.columns]
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

        df = df.rename(columns=HEADER_FIXUPS)
        df = df.rename(columns=COLUMN_RENAMES)
        by_year[year] = set(df.columns)

    all_cols = set.union(*by_year.values())
    common = set.intersection(*by_year.values())

    print(f"Common cols across years: {len(common)}")
    print(f"All cols across years:    {len(all_cols)}")

    for year in sorted(by_year):
        unique = sorted(by_year[year] - common)
        print(f"\nYear {year} unique cols ({len(unique)}):")
        for c in unique:
            print("  ", c)

    for year in sorted(by_year):
        missing = sorted(all_cols - by_year[year])
        print(f"\nYear {year} missing cols ({len(missing)}):")
        for c in missing:
            print("  ", c)


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--year", type=int, help="Year to normalize (e.g., 2023)")
    g.add_argument("--all", action="store_true", help="Normalize all years")
    g.add_argument("--diff-cols", action="store_true", help="Show column diffs across years")
    args = parser.parse_args()

    if args.diff_cols:
        diff_columns()
        return

    if args.all:
        for y in sorted(RAW_FILENAME_BY_YEAR.keys()):
            normalize_one_year(y)
    else:
        normalize_one_year(args.year)


if __name__ == "__main__":
    main()