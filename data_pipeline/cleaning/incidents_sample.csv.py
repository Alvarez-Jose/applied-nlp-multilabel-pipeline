import csv
from datetime import datetime, date
from typing import List

from data_pipeline.database.sheets_interface import create_incident
from utilities import validators
from data_pipeline import codebook


def parse_date(raw: str) -> date:
    """
    Parse RA 'Date of incident' like '1/2/2023' or '1/2/23' into a date object.
    """
    raw = raw.strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {raw!r}")


def infer_incident_type(row: dict) -> str:
    """
    Very simple rule-based incident_type using the RA row.
    You can improve this later.
    """
    desc = (row.get("Description of Violence") or "").strip()
    if desc:
        return desc

    parts: List[str] = []
    if validators.normalize_yes_no(row.get("Raid")) == "TRUE":
        parts.append("raid")
    if validators.normalize_yes_no(row.get("Arrest/Detention")) == "TRUE":
        parts.append("arrests")
    if row.get("Physical Assault"):
        parts.append(row["Physical Assault"])

    return " + ".join(parts) if parts else ""


def import_incidents_from_csv(csv_path: str) -> int:
    """
    Read an RA-coded CSV and append incidents into the Incidents sheet
    using the full Option B schema.
    """
    count = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            incident_id_raw = (row.get("ID") or "").strip()
            if not incident_id_raw:
                print("[WARN] Skipping row with empty ID.")
                continue

            # Governorate name and code
            gov_name = (row.get("Governorate") or "").strip()
            if not validators.validate_governorate(gov_name):
                print(f"[WARN] Unknown governorate: {gov_name!r} (ID={incident_id_raw})")

            governorate_code = incident_id_raw.split("-")[0] if "-" in incident_id_raw else ""
            if not governorate_code and gov_name in codebook.GOVERNORATE_CODES:
                governorate_code = codebook.GOVERNORATE_CODES[gov_name]

            # Date
            date_raw = (row.get("Date of incident") or "").strip()
            try:
                incident_date = parse_date(date_raw)
            except ValueError as e:
                print(f"[WARN] {e} (ID={incident_id_raw})")
                continue

            location_norm = (row.get("Location") or "").strip()

            incident_type = infer_incident_type(row)

            # Map RA columns to our schema fields
            jurisdiction = (row.get("Jurisdiction") or "").strip()
            location_type = (row.get("Location type") or "").strip()
            region = (row.get("Region") or "").strip()
            perpetrator = (row.get("Perpetrator") or "").strip()
            palestinian_involvement = (row.get("Palestinian involvement") or "").strip()

            raid = validators.normalize_yes_no(row.get("Raid"))
            arrest_detention = validators.normalize_yes_no(row.get("Arrest/Detention"))
            physical_assault = (row.get("Physical Assault") or "").strip()
            harm_to_property = (row.get("Harm to Property") or "").strip()
            dispossession = (row.get("Dispossession") or "").strip()
            religious_encroachment = validators.normalize_yes_no(row.get("Religious Encroachment"))
            restriction_of_freedoms = (row.get("Restriction of Freedoms") or "").strip()
            coercive_actions = (row.get("Coercive Actions") or "").strip()
            protest = validators.normalize_yes_no(row.get("Protest"))

            number_arrested = (row.get("Number Arrested") or "").strip()
            number_injured = (row.get("Number of Injuried") or "").strip()
            killed = (row.get("Killed") or "").strip()

            type_of_property_harmed = (row.get("Type of Property Harmed") or "").strip()
            type_of_property_dispossessed = (row.get("Type of Property Dispossessed") or "").strip()
            # For now, we don't have a separate "Material loss" column in the CSV
            material_loss = ""

            multi_community_incident = validators.normalize_yes_no(row.get("Multi-community Incident"))
            description = (row.get("Description of Violence") or "").strip()

            source_1 = (row.get("Source 1") or "").strip()
            source_2 = (row.get("Source 2") or "").strip()
            source_3 = (row.get("Source 3") or "").strip()
            source_4 = (row.get("Source 4") or "").strip()
            source_5 = (row.get("Source 5") or "").strip()
            source_6 = (row.get("Source 6") or "").strip()

            # Call your already-working Sheets helper
            create_incident(
                governorate_code=governorate_code,
                incident_date=incident_date,
                location_norm=location_norm,
                incident_type=incident_type,
                jurisdiction=jurisdiction,
                location_type=location_type,
                region=region,
                perpetrator=perpetrator,
                palestinian_involvement=palestinian_involvement,
                raid=raid,
                arrest_detention=arrest_detention,
                physical_assault=physical_assault,
                harm_to_property=harm_to_property,
                dispossession=dispossession,
                religious_encroachment=religious_encroachment,
                restriction_of_freedoms=restriction_of_freedoms,
                coercive_actions=coercive_actions,
                protest=protest,
                number_arrested=number_arrested,
                number_injured=number_injured,
                killed=killed,
                type_of_property_harmed=type_of_property_harmed,
                type_of_property_dispossessed=type_of_property_dispossessed,
                material_loss=material_loss,
                multi_community_incident=multi_community_incident,
                description=description,
                source_1=source_1,
                source_2=source_2,
                source_3=source_3,
                source_4=source_4,
                source_5=source_5,
                source_6=source_6,
            )
            count += 1

    print(f"Imported {count} incidents from {csv_path}")
    return count


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.cleaning.import_incidents_csv <path_to_csv>")
        raise SystemExit(1)

    import_incidents_from_csv(sys.argv[1])
