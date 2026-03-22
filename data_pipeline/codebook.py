"""
data_pipeline/codebook.py

Central place for ALL controlled vocabularies
based on the Oppression Database Coding Manual.
"""

# Governorates
GOVERNORATES = [
    "Nablus",
    "Qalqilya",
    "Tubas",
    "Salfit",
    "Tulkarm",
    "Jenin",
    "Jericho",
    "Jordan Valley",
    "Ramallah and al-Bireh",
    "Bethlehem",
    "Hebron",
    "Jerusalem",
]

# Governorate short codes (for IDs like HEB-20250312-01)
GOVERNORATE_CODES = {
    "Tulkarm": "TUL",
    "Qalqilya": "QAL",
    "Hebron": "HEB",
    "Salfit": "SAL",
    "Jenin": "JEN",
    "Bethlehem": "BET",
    "Ramallah and al-Bireh": "RAM",
    "Nablus": "NAB",
    "Tubas": "TUB",
    "Jericho": "JER",
    "Jordan Valley": "JVA",
    "Jerusalem": "JERU",
}

# Handle misspelled governorate names from RA sheets
GOVERNORATE_ALIASES = {
    "Qalqiliya": "Qalqilya",
    "Tulkarem": "Tulkarm",
}

# Standardized place name spellings (1/16/2026 — no apostrophes, capitalize all first letters)
# Keys are known variant spellings; values are canonical forms.
LOCATION_NAME_ALIASES = {
    # Tulkarm area
    "Tulkarem": "Tulkarm",
    # Bethlehem area
    "Beit Umar": "Beit Ummar",
    "Tuqu'": "Tuqu",
    "Tuqua": "Tuqu",
    "Nahalin": "Nahhalin",
    # Ramallah area
    "Umm Safa": "Um Safa",
    "Deir Debwan": "Deir Dibwan",
    "Turmusaya": "Turmus Ayya",
    "Jalazone": "Jalazun",
    # Nablus area
    "Burqin": "Birqin",
    "Burin": "Birin",
    "Jalboun": "Jalbun",
    # Jenin area
    "Tammoun": "Tammun",
    # Hebron area
    "Sammoa": "Sammou",
    "Ad-Dhahiriya": "A-Dhahiriya",
    # Jordan Valley
    "Al-Minya": "Al-Maniya",
    # Nablus / general
    "Nour Shams": "Nur Shams",
    # Salem
    "Salim": "Salem",
    # Bizzariya — no common variants but included for completeness
}

# Jurisdiction
JURISDICTIONS = [
    "Area A",
    "Area B",
    "Area C",
    "Mixed",
    "Unknown",
]

# Location Types
LOCATION_TYPES = [
    "Neighborhood",
    "Village",
    "Town",
    "City",
    "Refugee Camp",
    "Community",  # for Masafer Yatta cluster
]

# Perpetrator labels
PERPETRATORS = [
    "Israeli Police",
    "Israeli Settlers",
    "Israeli Soldiers",
    "Israeli Citizens",
    "Palestinian Authority",
]

# Palestinian Involvement
PALESTINIAN_INVOLVEMENT = [
    "Armed Individuals",
    "Militant Groups",
    "Civilians",
    "None/Not Specified",
]

# Harm to Property Types
HARM_TO_PROPERTY_TYPES = [
    "Vandalism",
    "Destruction",
]

# Type of Property Harmed / Dispossessed
PROPERTY_TYPES = [
    "Personal objects",
    "Artifacts",
    "Agricultural",
    "Natural resource",
    "Religious",
    "Livestock",
    "Medical Facilities/Supplies",
    "Businesses",
    "Public Buildings",
    "Infrastructure",
    "Residences",
    "Automobiles/Vehicles",
]

# Dispossession Types
DISPOSSESSION_TYPES = [
    "Attempted Theft/Seizure",   # includes confiscation orders not yet executed
    "Requisition",
    "Temporary Requisition",
    "Theft/Seizure",
    "Killing Animals",
]

# Physical Assault Types
PHYSICAL_ASSAULT_TYPES = [
    "Airborne",
    "Beating",
    "Tear Gas",
    "Shooting",
    "Unspecified Assault",
    "Stun Grenades",
    "Vehicular",
    "Dogs",
    "Stones",
    "Human Shield",
    "Pepper Spray",
    "Stabbing",
]

# Restriction of Freedoms Types
RESTRICTION_OF_FREEDOMS_TYPES = [
    "Curfew",
    "Checkpoint",
    "Roadblock",
    "Barriers",
    "Prevention of Working",
    "Outpost",
    "Prevention of Aid",
    "Prevention of Religious Practices",
]

# Coercive Actions Types
COERCIVE_ACTION_TYPES = [
    "Intimidation",
    "Verbal Harassment",
    "Search/Interrogation",
    "Deployment of Troops",
    "Distribution of Demolition Orders",
]

# Binary event categories (Yes/No)
BINARY_EVENT_FLAGS = [
    "Raid",
    "Arrest/Detention",
    "Physical Assault",
    "Restriction of Freedoms",
    "Coercive Actions",
    "Religious Encroachment",
    "Harm to Property",
    "Dispossession",
    "Multi-community Incident",
    "Resistance and Collective Action",  # model column: protest_y
]
