from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional


@dataclass
class Report:
    '''
    One source account(article, NGO report, etc)
    '''
    source_name: str
    url: str
    title: str
    body: str
    
    published_at_utc: datetime
    published_at_local: datetime
    
    language: str
    location_raw: Optional[str] = None
    actors_raw: Optional[str] = None
    
@dataclass
class Incident:
    '''One *real world event* that multiple reports may describe - we can keep this minimal at first - Gives unique incident'''
    incident_id: int # HEB-20250312-01
    governorate_code: str  # HEB
    incident_date: date # local Palestine data
    location_norm: str  # Herbon, Old city
    incident_type: Optional[str] = None