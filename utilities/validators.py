from typing import Optional
from data_pipeline import codebook


# Normalize Yes/No values into Google Sheets style TRUE/FALSE
def normalize_yes_no(value: Optional[str]) -> str:
    if value is None:
        return "FALSE"
    v = str(value).strip().lower()
    if v in ("yes", "true", "y", "1"):
        return "TRUE"
    if v in ("no", "false", "n", "0"):
        return "FALSE"
    return "FALSE"  # default for ambiguous cases


# Governorate validation
def validate_governorate(name: str) -> bool:
    if not name:
        return False
    return name in codebook.GOVERNORATES


# Location type validation
def validate_location_type(loc_type: str) -> bool:
    if not loc_type:
        return False
    return loc_type in codebook.LOCATION_TYPES


# Jurisdiction validation (Area A/B/C/etc.)
def validate_jurisdiction(jur: str) -> bool:
    if not jur:
        return True  # allow missing jurisdiction (e.g., Jerusalem)
    return jur in codebook.JURISDICTIONS


# Perpetrator validation
def validate_perpetrator(perp: str) -> bool:
    if not perp:
        return True  # allow empty
    return perp in codebook.PERPETRATORS


# Palestinian involvement validation
def validate_palestinian_involvement(label: str) -> bool:
    if not label:
        return True  # allow empty
    return label in codebook.PALESTINIAN_INVOLVEMENT


# Harm to Property (subcategory)
def validate_harm_to_property_type(prop_type: str) -> bool:
    if not prop_type:
        return True  # allow empty
    return prop_type in codebook.HARM_TO_PROPERTY_TYPES


# Dispossession type validation
def validate_dispossession_type(label: str) -> bool:
    if not label:
        return True
    return label in codebook.DISPOSSESSION_TYPES


# Physical assault subtype validation
def validate_physical_assault_type(label: str) -> bool:
    if not label:
        return True
    return label in codebook.PHYSICAL_ASSAULT_TYPES


# Restriction of freedoms subtype validation
def validate_restriction_of_freedoms(label: str) -> bool:
    if not label:
        return True
    return label in codebook.RESTRICTION_OF_FREEDOMS_TYPES


# Coercive actions subtype validation
def validate_coercive_action(label: str) -> bool:
    if not label:
        return True
    return label in codebook.COERCIVE_ACTION_TYPES


# Property type validation (what was harmed/dispossessed)
def validate_property_type(label: str) -> bool:
    if not label:
        return True
    return label in codebook.PROPERTY_TYPES

