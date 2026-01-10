from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Report:
    """
    One source account (article, NGO report, etc.)
    RAs or scrapers create these. They link to an Incident.
    Date-only version for simplicity.
    """
    source_name: str
    url: str
    title: str
    body: str

    published_date: date  # Date only, no time

    language: str
    location_raw: Optional[str] = None
    actors_raw: Optional[str] = None


@dataclass
class Incident:
    """
    One real-world event. Multiple reports may describe it.
    Mirrors the Google Sheet + codebook structure.
    """

    # Core identifiers
    incident_id: str               # e.g., "HEB-20250312-01"
    governorate_code: str          # e.g., "HEB"
    incident_date: date            # Palestine local date
    location_norm: str             # normalized location (e.g., "Hebron, Old City")

    # Incident classification (codebook)
    incident_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    location_type: Optional[str] = None
    perpetrator: Optional[str] = None
    palestinian_involvement: Optional[str] = None

    # Event tags (boolean or string values)
    raid: Optional[str] = None
    arrest_detention: Optional[str] = None
    physical_assault: Optional[str] = None
    harm_to_property: Optional[str] = None
    dispossession: Optional[str] = None
    religious_encroachment: Optional[str] = None
    restriction_of_freedoms: Optional[str] = None
    coercive_actions: Optional[str] = None
    protest: Optional[str] = None

    # Numeric fields
    number_arrested: Optional[str] = None
    number_injured: Optional[str] = None
    killed: Optional[str] = None

    # Description
    description: Optional[str] = None

    # Multi-community indicator
    multi_community_incident: Optional[str] = None

    # Source fields (we support up to 6)
    source_1: Optional[str] = None
    source_2: Optional[str] = None
    source_3: Optional[str] = None
    source_4: Optional[str] = None
    source_5: Optional[str] = None
    source_6: Optional[str] = None

    # Property-specific fields
    type_of_property_harmed: Optional[str] = None
    type_of_property_dispossessed: Optional[str] = None
    material_loss: Optional[str] = None
