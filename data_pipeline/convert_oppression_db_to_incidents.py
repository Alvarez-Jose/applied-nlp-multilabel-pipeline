"""
data_pipeline/convert_oppression_db_to_incidents.py

Converts any raw Oppression Database CSV into a normalized, training-ready
processed CSV with enriched descriptions.

Core value:
    Many raw rows have very short descriptions (e.g. "Settlers vandalize house").
    This script generates a full structured description from the coded fields,
    either replacing short descriptions or supplementing them.

Output is written directly to data/processed/ in canonical schema format,
bypassing normalize_raw.py (all normalization is done inline). The output
file is auto-detected by export_training_data.py.

Workflow:
    python data_pipeline/convert_oppression_db_to_incidents.py --year 2024
    python modeling/training/export_training_data.py

Usage:
    # Convert 2024 (default):
    python data_pipeline/convert_oppression_db_to_incidents.py

    # Convert a specific year:
    python data_pipeline/convert_oppression_db_to_incidents.py --year 2023

    # Custom input/output:
    python data_pipeline/convert_oppression_db_to_incidents.py \\
        --input data/raw/my_dataset.csv \\
        --output data/processed/my_dataset_converted.csv

    # Replace ALL descriptions (even long ones):
    python data_pipeline/convert_oppression_db_to_incidents.py --overwrite-descriptions

    # Preview stats without writing:
    python data_pipeline/convert_oppression_db_to_incidents.py --dry-run

Description generation strategy:
    - If the existing description is >= --min-desc-len chars (default 50):
        keep as-is
    - Otherwise: generate from structured fields and prepend the original
        as a suffix (so no information is lost)

Requirements:
    pip install pandas
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.normalize_raw import HEADER_FIXUPS, COLUMN_RENAMES, CANONICAL_OUTPUT_COLS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Default raw files per year (matches normalize_raw.py)
RAW_FILENAME_BY_YEAR = {
    2023: "Oppression database - 2023.csv",
    2024: "Oppression database - 2024.csv",
    2025: "Oppression database - 2025.csv",
}

# Boolean-style labels (True/False, yes/no, 0/1)
BOOL_LABELS = {
    "raid", "arrest_detention", "religious_encroachment",
    "protest", "multi_community_incident",
}

# Presence-style labels (empty = negative, any string = positive + type info)
PRESENCE_LABELS = {
    "physical_assault", "harm_to_property", "dispossession",
    "coercive_actions", "restriction_of_freedoms",
}


# ===========================================================================
# Helpers
# ===========================================================================

def is_positive(val) -> bool:
    """Return True if val represents a positive / present value."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val) and not pd.isna(val)
    s = str(val).strip().lower()
    return s not in {"", "0", "false", "no", "n", "f", "nan", "none"}


def coerce_str(val) -> str:
    """Return a clean string or empty string for missing/falsy values."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return s if s.lower() not in {"nan", "none"} else ""


# ===========================================================================
# Description builder
# ===========================================================================

def build_description(row: pd.Series) -> str:
    """
    Generate a natural language description from structured incident fields.

    The description covers:
      - Who (perpetrator) and where (location + governorate + jurisdiction)
      - What actions occurred (each label that is positive)
      - Counts (arrested, injured, killed)
      - Property details (type of harm, dispossession subtype)
      - Multi-community and protest flags

    Fields used (canonical names, post-normalization):
        perpetrator, governorate, location_name, location_type, jurisdiction,
        area, raid, arrest_detention, number_arrested, physical_assault,
        killed, number_injured, coercive_actions, restriction_of_freedoms,
        restriction_of_freedoms_mechanisms, religious_encroachment,
        harm_to_property, type_of_property_harmed, dispossession,
        type_of_property_dispossessed, multi_community_incident, protest,
        type_of_protest
    """
    parts = []

    # --- Location context ---
    perpetrator    = coerce_str(row.get("perpetrator"))
    governorate    = coerce_str(row.get("governorate"))
    location_name  = coerce_str(row.get("location_name"))
    location_type  = coerce_str(row.get("location_type")).rstrip()
    jurisdiction   = coerce_str(row.get("jurisdiction"))

    loc_parts = []
    if location_name:
        loc_parts.append(location_name)
    if location_type and location_type.lower() not in location_name.lower():
        loc_parts.append(location_type.lower())
    if governorate:
        loc_parts.append(f"{governorate} governorate")
    location_str = ", ".join(loc_parts) if loc_parts else "the West Bank"

    jur_str = f" ({jurisdiction})" if jurisdiction else ""
    perp_str = perpetrator if perpetrator else "Israeli forces"

    # --- Opening sentence (raid or generic) ---
    if is_positive(row.get("raid")):
        parts.append(
            f"{perp_str} conducted a raid in {location_str}{jur_str}."
        )
    else:
        parts.append(
            f"An incident involving {perp_str} was recorded in {location_str}{jur_str}."
        )

    # --- Arrests ---
    if is_positive(row.get("arrest_detention")):
        n = coerce_str(row.get("number_arrested"))
        if n and n not in {"0"}:
            parts.append(f"{n} Palestinian(s) were arrested or detained.")
        else:
            parts.append("Palestinians were arrested or detained.")

    # --- Physical assault ---
    if is_positive(row.get("physical_assault")):
        assault_type = coerce_str(row.get("physical_assault"))
        type_str = (
            f" ({assault_type})"
            if assault_type and assault_type.lower() not in {"yes", "true", "1"}
            else ""
        )
        parts.append(f"Physical assault was reported{type_str}.")

        n_inj = coerce_str(row.get("number_injured"))
        if n_inj and n_inj not in {"0"}:
            parts.append(f"{n_inj} person(s) were injured.")

    # --- Killed ---
    killed = coerce_str(row.get("killed"))
    if killed and killed not in {"0"}:
        parts.append(f"{killed} Palestinian(s) were killed.")

    # --- Harm to property ---
    if is_positive(row.get("harm_to_property")):
        harm_type  = coerce_str(row.get("harm_to_property"))
        prop_type  = coerce_str(row.get("type_of_property_harmed"))
        type_str = (
            f" ({harm_type})"
            if harm_type and harm_type.lower() not in {"yes", "true", "1"}
            else ""
        )
        prop_str = f" of {prop_type}" if prop_type else ""
        parts.append(f"Property damage{type_str}{prop_str} was reported.")

    # --- Dispossession ---
    if is_positive(row.get("dispossession")):
        disp_type  = coerce_str(row.get("dispossession"))
        prop_disp  = coerce_str(row.get("type_of_property_dispossessed"))
        type_str = (
            f" ({disp_type})"
            if disp_type and disp_type.lower() not in {"yes", "true", "1"}
            else ""
        )
        prop_str = f" of {prop_disp}" if prop_disp else ""
        parts.append(f"Property dispossession{type_str}{prop_str} was reported.")

    # --- Religious encroachment ---
    if is_positive(row.get("religious_encroachment")):
        parts.append("Religious encroachment was reported.")

    # --- Restriction of freedoms ---
    if is_positive(row.get("restriction_of_freedoms")):
        mech = coerce_str(row.get("restriction_of_freedoms_mechanisms"))
        mech_str = f" ({mech})" if mech else ""
        parts.append(f"Restrictions on freedom of movement were imposed{mech_str}.")

    # --- Coercive actions ---
    if is_positive(row.get("coercive_actions")):
        ca = coerce_str(row.get("coercive_actions"))
        type_str = (
            f" ({ca})"
            if ca and ca.lower() not in {"yes", "true", "1"}
            else ""
        )
        parts.append(f"Coercive actions were reported{type_str}.")

    # --- Protest ---
    if is_positive(row.get("protest")):
        p_type = coerce_str(row.get("type_of_protest"))
        type_str = f" ({p_type})" if p_type else ""
        parts.append(f"A Palestinian protest occurred{type_str}.")

    # --- Multi-community ---
    if is_positive(row.get("multi_community_incident")):
        parts.append("The incident affected multiple communities.")

    return " ".join(parts)


# ===========================================================================
# Normalization helpers (replicate normalize_raw.py inline)
# ===========================================================================

BOOL_MAP = {
    "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
    "no": 0, "n": 0, "false": 0, "f": 0, "0": 0,
}


def _normalize_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize boolean-ish columns to 0/1 Int64."""
    bool_cols = BOOL_LABELS  # canonical names
    for col in bool_cols:
        if col not in df.columns:
            continue
        def _map(v):
            if pd.isna(v):
                return pd.NA
            return BOOL_MAP.get(str(v).strip().lower(), pd.NA)
        df[col] = df[col].map(_map).astype("Int64")
    return df


def _clean_header(col: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(col).strip())


# ===========================================================================
# Main conversion
# ===========================================================================

def convert(
    input_path: Path,
    output_path: Path,
    year: int,
    min_desc_len: int,
    overwrite_descriptions: bool,
    dry_run: bool,
) -> None:

    # --- 1. Load raw CSV ---
    log.info("Reading %s ...", input_path)
    df = pd.read_csv(input_path, dtype=str).fillna("")

    # --- 2. Apply same header fixups as normalize_raw.py ---
    df.columns = [_clean_header(c) for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.rename(columns=HEADER_FIXUPS)
    df = df.rename(columns=COLUMN_RENAMES)

    log.info("Loaded %d rows, %d columns after rename.", len(df), len(df.columns))

    # --- 3. Normalize booleans ---
    df = _normalize_booleans(df)

    # --- 4. Deduplicate ---
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log.warning("Dropped %d exact duplicate rows.", before - len(df))

    # --- 5. Build row_id ---
    if "id" in df.columns:
        id_key = df["id"].fillna("MISSING_ID").astype(str)
        suffix = df.groupby(id_key).cumcount().astype(str)
        df["row_id"] = (id_key + "-" + suffix).astype(str)
    else:
        df["row_id"] = df.index.astype(str)

    # --- 6. Generate / enrich descriptions ---
    if "description" not in df.columns:
        df["description"] = ""

    generated = 0
    supplemented = 0

    def _resolve_description(row: pd.Series) -> str:
        original = coerce_str(row.get("description"))
        if overwrite_descriptions or len(original) < min_desc_len:
            generated_desc = build_description(row)
            if original and not overwrite_descriptions:
                # Prepend the generated text; keep original as a trailing note
                return f"{generated_desc} [Source note: {original}]"
            return generated_desc
        return original

    new_descs = []
    for _, row in df.iterrows():
        original = coerce_str(row.get("description"))
        new_desc = _resolve_description(row)
        new_descs.append(new_desc)
        if not original:
            generated += 1
        elif new_desc != original:
            supplemented += 1

    df["description"] = new_descs

    # --- 7. Add year / source_file columns ---
    df["year"] = year
    df["source_file"] = output_path.name

    # --- 8. Enforce canonical output columns ---
    for col in CANONICAL_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[CANONICAL_OUTPUT_COLS]

    # --- 9. Summary ---
    desc_lengths = df["description"].str.len()
    label_rates = {}
    for col in list(BOOL_LABELS) + list(PRESENCE_LABELS):
        if col in df.columns:
            label_rates[col] = is_positive
            rate = df[col].apply(is_positive).mean()
            label_rates[col] = rate

    print("\n" + "=" * 60)
    print("  CONVERSION SUMMARY")
    print("=" * 60)
    print(f"  Input              : {input_path}")
    print(f"  Year               : {year}")
    print(f"  Rows               : {len(df)}")
    print(f"  Descriptions kept  : {len(df) - generated - supplemented}")
    print(f"  Descriptions supplemented: {supplemented}")
    print(f"  Descriptions generated   : {generated}")
    print(f"  Desc length  min/mean/max: "
          f"{desc_lengths.min():.0f} / {desc_lengths.mean():.0f} / {desc_lengths.max():.0f}")
    print()
    print("  Label positive rates:")
    for col, rate in label_rates.items():
        print(f"    {col:35s} {rate*100:5.1f}%")
    print()
    print(f"  Output: {output_path}")
    print("=" * 60 + "\n")

    if dry_run:
        log.info("--dry-run: no file written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Wrote %d rows to %s", len(df), output_path)


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a raw Oppression Database CSV into a normalized, "
            "training-ready processed CSV with enriched descriptions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        default=2024,
        choices=[2023, 2024, 2025],
        help="Year of the dataset. Used to select the default input file (default: 2024).",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help=(
            "Path to the raw CSV. "
            "Default: data/raw/Oppression database - {year}.csv"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Output path for the processed CSV. "
            "Default: data/processed/oppression_{year}_converted.csv"
        ),
    )
    parser.add_argument(
        "--min-desc-len",
        type=int,
        default=50,
        help=(
            "Descriptions shorter than this (chars) are replaced or supplemented "
            "with generated text. Default: 50."
        ),
    )
    parser.add_argument(
        "--overwrite-descriptions",
        action="store_true",
        help="Replace ALL descriptions with generated text, even long ones.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the summary but do not write the output file.",
    )
    args = parser.parse_args()

    input_path = (
        Path(args.input)
        if args.input
        else PROJECT_ROOT / "data" / "raw" / RAW_FILENAME_BY_YEAR[args.year]
    )
    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data" / "processed" / f"oppression_{args.year}_converted.csv"
    )

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    convert(
        input_path=input_path,
        output_path=output_path,
        year=args.year,
        min_desc_len=args.min_desc_len,
        overwrite_descriptions=args.overwrite_descriptions,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
