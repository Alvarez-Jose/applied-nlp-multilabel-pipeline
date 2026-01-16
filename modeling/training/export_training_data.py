import re
from pathlib import Path
from typing import List

import pandas as pd


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "modeling" / "training" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = [
    "Oppression database - 2023.csv",
    "Oppression database - 2024.csv",
    "Oppression database - 2025.csv",
]


# ---------- Column canonicalization ----------
def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Clean headers
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed") or c in ("...", "")],
        errors="ignore",
    )

    # 2) Rename known variants → canonical names
    rename = {
        "ID": "incident_id",
        "Date of incident": "incident_date",

        # location/region variants
        "Region": "region",
        "Location (WB or Gaza)": "region",
        "Location": "location",
        "Location name": "location",
        "Location type": "location_type",
        "Location Type": "location_type",

        "Governorate": "governorate",
        "Jurisdiction": "jurisdiction",
        "Jurisdiction ": "jurisdiction",

        "Perpetrator": "perpetrator",
        "Perpetrator ": "perpetrator",

        "Palestinian involvement": "palestinian_involvement",
        "Palestinian involvement ": "palestinian_involvement",

        # manual labels / fields
        "Raid": "raid",
        "Arrest/Detention": "arrest_detention",
        "Number Arrested": "number_arrested",
        "Physical Assault": "physical_assault",
        "Physical Assault ": "physical_assault",
        "Killed": "killed",
        "Number of Injuried": "number_injured",
        "Number of Injuried ": "number_injured",
        "Coercive Actions": "coercive_actions",
        "Restriction of Freedoms": "restriction_of_freedoms",
        "Religious Encroachment": "religious_encroachment",
        "Harm to Property": "harm_to_property",
        "Harm to Property ": "harm_to_property",
        "Dispossession": "dispossession",
        "Disposession": "dispossession",

        "Type of Property Harmed": "type_of_property_harmed",
        "Type of Property Dispossessed": "type_of_property_dispossessed",
        "Type of Propery Disposessed": "type_of_property_dispossessed",

        "Description of Violence": "description",
        "Multi-community Incident": "multi_community_incident",
        "Multi-Community Incident": "multi_community_incident",
        "Protest": "protest",
        "Type of Protest": "type_of_protest",
        "Type of protest": "type_of_protest",

        "Property": "property_detail",
        "Property ": "property_detail",
    }

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # 3) Sources
    for i in range(1, 7):
        src = f"Source {i}"
        if src in df.columns:
            df = df.rename(columns={src: f"source_{i}"})

    return df


# ---------- Target normalization ----------
def norm_bool_true_false(x) -> int:
    """
    For TRUE/FALSE style fields.
    Treats missing/blank as 0 (not present). If you later want "unknown",
    we can switch to tri-state labels.
    """
    if pd.isna(x):
        return 0
    s = str(x).strip().upper()
    if s == "":
        return 0
    return 1 if s in ("TRUE", "YES", "Y", "1") else 0


def norm_presence(x) -> int:
    """
    For fields where presence is represented by a non-empty string
    (e.g., physical_assault="Shooting", harm_to_property="Vandalism").
    Any non-empty and non-FALSE/0 value counts as present.
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

    for fname in RAW_FILES:
        path = RAW_DATA / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing raw CSV: {path}")

        df = pd.read_csv(path, encoding="utf-8-sig")
        df = canonicalize_columns(df)

        # year tag from filename
        m = re.search(r"(\d{4})", fname)
        df["year"] = int(m.group(1)) if m else None
        df["source_file"] = fname

        frames.append(df)

    full = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure required columns exist (create empty if missing)
    required_text_cols = [
        "incident_id", "incident_date", "description",
        "region", "location", "location_type", "governorate", "jurisdiction",
        "perpetrator", "palestinian_involvement",
    ]
    for c in required_text_cols:
        if c not in full.columns:
            full[c] = ""

    # Manual-aligned binary targets
    # TRUE/FALSE-style
    for col in ["raid", "arrest_detention", "religious_encroachment", "protest", "multi_community_incident"]:
        if col not in full.columns:
            full[col] = ""
        full[col + "_y"] = full[col].apply(norm_bool_true_false)

    # presence-style (string-valued)
    for col in ["physical_assault", "coercive_actions", "restriction_of_freedoms", "harm_to_property", "dispossession"]:
        if col not in full.columns:
            full[col] = ""
        full[col + "_y"] = full[col].apply(norm_presence)

    # Normalize incident_date to string (keep raw; we can parse later if needed)
    full["incident_date"] = full["incident_date"].astype(str).str.strip()

    # Keep columns for modeling + auditing
    keep_cols = [
        "incident_id", "incident_date", "year", "source_file",
        "region", "location", "location_type", "governorate", "jurisdiction",
        "perpetrator", "palestinian_involvement",
        "description",

        # raw subtype-ish fields (useful later)
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


# ---------- Main ----------
def main():
    df = build()

    # Save canonical dataset
    out_parquet = OUT_DIR / "incidents_2023_2025.parquet"
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception as e:
        print(f"[WARN] Failed to write parquet ({e}). Install pyarrow: pip install pyarrow")
        # Still continue with CSV outputs

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
    print("Raw dir:", RAW_DATA)
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