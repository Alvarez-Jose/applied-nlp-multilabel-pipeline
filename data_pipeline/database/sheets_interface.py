import time
from gspread import Client
from google.oauth2.service_account import Credentials
from gspread.utils import ValueInputOption

from datetime import date
from typing import List, Dict, Optional, Set

import gspread
import pandas as pd

from data_pipeline.database.models import Report
from utilities.incident_id import format_incident_id

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Path to the service account JSON (located in project root)
SERVICE_ACCOUNT_FILE = "service_account.json"

# Spreadsheet ID (long string in the Google Sheet URL)
SPREADSHEET_ID = "1Z7zu2JLxOIU1yK3SrXIz8yN3a-2Kzd7d0g0-VNrkwaI"

INCIDENTS_SHEET_NAME = "Incidents"
REPORTS_SHEET_NAME   = "Reports"
REVIEW_SHEET_NAME    = "Review"

# Maximum retries for ID collision
MAX_ID_RETRIES = 20

# Governorate code → full name (must match actual sheet values)
GOVERNORATE_CODE_TO_NAME = {
    "NAB":  "Nablus",
    "QAL":  "Qalqilya",
    "TUB":  "Tubas",
    "SAL":  "Salfit",
    "TUL":  "Tulkarm",
    "JEN":  "Jenin",
    "JER":  "Jericho",
    "JVA":  "Jordan Valley",
    "RAM":  "Ramallah and al-Bireh",
    "BET":  "Bethlehem",
    "HEB":  "Hebron",
    "JERU": "Jerusalem",
}


def get_client() -> Client:
    """Authorize and return a gspread client using the service account."""
    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    client: Client = gspread.authorize(creds)
    return client


def get_sheet(sheet_name: str):
    """Get a worksheet object by name."""
    client = get_client()
    sh = client.open_by_key(SPREADSHEET_ID)
    return sh.worksheet(sheet_name)


def list_incidents() -> List[Dict]:
    """
    Return all incidents as a list of dicts using the Incidents sheet header row.

    Uses get_all_values() instead of get_all_records() so that duplicate column
    headers in the sheet do not raise an error. Duplicate headers are suffixed
    with _2, _3, etc. to keep all columns accessible.
    """
    ws = get_sheet(INCIDENTS_SHEET_NAME)
    all_values = ws.get_all_values()
    if not all_values:
        return []

    # Deduplicate headers: if a name appears more than once, suffix later copies
    raw_headers = all_values[0]
    headers = []
    seen: Dict[str, int] = {}
    for h in raw_headers:
        if h in seen:
            seen[h] += 1
            headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 1
            headers.append(h)

    return [dict(zip(headers, row)) for row in all_values[1:]]


def report_url_exists(url: str) -> bool:
    """
    Check if a report with the given URL already exists in the Reports sheet.
    
    Args:
        url: The URL to check
    
    Returns:
        True if URL exists, False otherwise
    """
    ws = get_sheet(REPORTS_SHEET_NAME)
    
    # Assuming URL is column D (4th column)
    # Get all values from column D (skip header)
    all_urls = ws.col_values(4)[1:]  # Skip header row
    
    return url in all_urls
# In sheets_interface.py, add caching
_url_cache = None
_url_cache_time = None

def get_all_existing_urls() -> Set[str]:
    """Get all URLs from Reports sheet once, with caching."""
    global _url_cache, _url_cache_time
    
    # Cache for 5 minutes
    if _url_cache and _url_cache_time and (time.time() - _url_cache_time < 300):
        return _url_cache.copy()
    
    ws = get_sheet(REPORTS_SHEET_NAME)
    all_urls = ws.col_values(4)[1:]
    
    _url_cache = {
        str(url).strip() 
        for url in all_urls 
        if url is not None and str(url).strip()
    }
    _url_cache_time = time.time()
    
    return _url_cache.copy()

def get_next_sequence(governorate_code: str, incident_date: date) -> int:
    """Count how many incidents already exist for this governorate + date, then assign the next sequence number."""
    incidents = list_incidents()
    target_date_str = incident_date.isoformat()
    # The sheet stores full governorate name in the 'governorate' column
    gov_name = GOVERNORATE_CODE_TO_NAME.get(governorate_code.upper(), governorate_code)

    count = sum(
        1
        for inc in incidents
        if inc.get("governorate") == gov_name
        and inc.get("date") == target_date_str
    )

    return count + 1


def append_row_by_header(ws, data: dict) -> None:
    """Append a row ensuring proper column mapping regardless of order."""
    headers = ws.row_values(1)
    row = [data.get(h, "") for h in headers]
    ws.append_row(row, value_input_option=ValueInputOption.raw)


def build_row_from_headers(headers: List[str], data: dict) -> List:
    """Build a row in the correct column order based on sheet headers."""
    return [data.get(h, "") for h in headers]


def create_incident(
    governorate_code: str,
    incident_date: date,
    location_norm: str,
    incident_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    location_type: Optional[str] = None,
    region: Optional[str] = None,
    perpetrator: Optional[str] = None,
    palestinian_involvement: Optional[str] = None,
    raid: Optional[str] = None,
    arrest_detention: Optional[str] = None,
    physical_assault: Optional[str] = None,
    harm_to_property: Optional[str] = None,
    dispossession: Optional[str] = None,
    religious_encroachment: Optional[str] = None,
    restriction_of_freedoms: Optional[str] = None,
    coercive_actions: Optional[str] = None,
    protest: Optional[str] = None,
    number_arrested: Optional[str] = None,
    number_injured: Optional[str] = None,
    killed: Optional[str] = None,
    type_of_property_harmed: Optional[str] = None,
    type_of_property_dispossessed: Optional[str] = None,
    material_loss: Optional[str] = None,
    multi_community_incident: Optional[str] = None,
    description: Optional[str] = None,
    source_1: Optional[str] = None,
    source_2: Optional[str] = None,
    source_3: Optional[str] = None,
    source_4: Optional[str] = None,
    source_5: Optional[str] = None,
    source_6: Optional[str] = None,
) -> str:
    """
    Create a new incident row in the Incidents sheet and return its incident_id.

    Only the first three arguments are required; everything else is optional and
    defaults to empty string in the sheet if not provided.
    
    Includes retry logic for ID collisions.
    """
    ws = get_sheet(INCIDENTS_SHEET_NAME)

    # Validate required headers exist (using actual sheet column names)
    headers = ws.row_values(1)
    required_headers = {"incident_id", "date", "governorate", "description"}
    missing_headers = required_headers - set(headers)
    if missing_headers:
        raise RuntimeError(f"Incidents sheet missing required headers: {missing_headers}")

    # Get all existing IDs once to avoid multiple sheet reads
    existing_ids = set(ws.col_values(1))  # incident_id column

    # Calculate base sequence and try incrementing for collisions
    base_seq = get_next_sequence(governorate_code, incident_date)
    incident_id = None

    for attempt in range(MAX_ID_RETRIES):
        seq = base_seq + attempt
        candidate_id = format_incident_id(governorate_code, incident_date, seq)
        if candidate_id not in existing_ids:
            incident_id = candidate_id
            break

    if incident_id is None:
        raise RuntimeError(f"Failed to allocate unique incident_id after {MAX_ID_RETRIES} attempts")

    # Resolve full governorate name (sheet stores full name, not code)
    gov_name = GOVERNORATE_CODE_TO_NAME.get(governorate_code.upper(), governorate_code)

    # Combine source URLs into a single 'source' column (comma-separated)
    source_parts = [s for s in [source_1, source_2, source_3, source_4, source_5, source_6]
                    if s and s.strip()]
    source_combined = ", ".join(source_parts)

    # Build data dict using actual Incidents sheet column names
    data = {
        "incident_id":    incident_id,
        "date":           incident_date.isoformat(),
        "governorate":    gov_name,
        "area":           region or "",
        "location_name":  location_norm,
        "location_type":  location_type or "",
        "perpetrator":    perpetrator or "",
        "description":    description or "",
        "raid":                   raid or "",
        "arrest_detention":       arrest_detention or "",
        "physical_assault":       physical_assault or "",
        "harm_to_property":       harm_to_property or "",
        "dispossession":          dispossession or "",
        "religious_encroachment": religious_encroachment or "",
        "restriction_of_freedoms": restriction_of_freedoms or "",
        "coercive_actions":       coercive_actions or "",
        "protest":                protest or "",
        "multi_community_incident": multi_community_incident or "",
        "review_status":   "",
        "reviewer_notes":  "",
        "source":          source_combined,
        "url":             source_1 or "",   # primary URL in dedicated column
    }

    # Use append_row_by_header for safe column mapping
    append_row_by_header(ws, data)
    return incident_id


def add_reports(reports: List[Report]) -> None:
    """
    Append a batch of reports to the Reports sheet (date-only version).
    
    Header for Reports MUST be updated to:
    report_id | incident_id | source_name | url | title | body_text |
    published_date | language | location_raw | actors_raw
    
    (Removed published_at_utc and published_at_local, added published_date)
    """
    ws = get_sheet(REPORTS_SHEET_NAME)

    # Get headers once
    headers = ws.row_values(1)

    # Derive next report_id from the max existing ID (safe against row deletions)
    existing_ids = ws.col_values(1)[1:]  # skip header
    max_num = 0
    for rid in existing_ids:
        rid = str(rid).strip()
        if rid.startswith("R") and rid[1:].isdigit():
            max_num = max(max_num, int(rid[1:]))

    rows_to_append = []
    for i, r in enumerate(reports):
        report_number = max_num + i + 1
        report_id = f"R{report_number:05d}"     # e.g. R00001

        # Format date as ISO string (YYYY-MM-DD)
        published_date_str = r.published_date.isoformat()

        # Handle optional fields (convert None to empty string)
        location_raw = r.location_raw or ""
        actors_raw = r.actors_raw or ""

        # Convert to dict for proper column mapping
        data = {
            "report_id": report_id,
            "incident_id": "",  # incident_id (future: linking logic)
            "source_name": r.source_name or "",
            "url": r.url or "",
            "title": r.title or "",
            "body_text": r.body or "",
            "published_date": published_date_str,  # Date only (ISO format)
            "language": r.language or "",
            "location_raw": location_raw,
            "actors_raw": actors_raw,
        }

        # Build row in correct column order
        row = build_row_from_headers(headers, data)
        rows_to_append.append(row)

    # Append all rows in one API call
    ws.append_rows(rows_to_append, value_input_option=ValueInputOption.raw)


def push_review_batch(df: pd.DataFrame) -> int:
    """
    Append new prediction rows to the 'Review' sheet, skipping row_ids already present.

    If the sheet is empty, writes column headers first (taken from df.columns).
    If the sheet already has rows, columns are aligned to the existing headers;
    DataFrame columns not in the sheet are dropped; sheet columns not in the
    DataFrame are written as empty strings.

    Returns the number of rows actually appended (0 if all already present).
    """
    ws = get_sheet(REVIEW_SHEET_NAME)
    all_values = ws.get_all_values()

    if not all_values:
        # Empty sheet — write header row first
        headers = list(df.columns)
        ws.append_row(headers, value_input_option=ValueInputOption.raw)
        existing_row_ids: Set[str] = set()
    else:
        headers = all_values[0]
        try:
            rid_idx = headers.index("row_id")
            existing_row_ids = {
                row[rid_idx]
                for row in all_values[1:]
                if len(row) > rid_idx and row[rid_idx]
            }
        except ValueError:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Review sheet has no 'row_id' column — cannot check for duplicates. "
                "All rows will be appended."
            )
            existing_row_ids = set()

    # Drop rows already in the sheet
    if "row_id" in df.columns:
        new_df = df[~df["row_id"].astype(str).isin(existing_row_ids)].copy()
    else:
        new_df = df.copy()

    if new_df.empty:
        return 0

    # Build rows aligned to sheet header order
    rows: List[List] = []
    for _, row in new_df.iterrows():
        row_data = []
        for h in headers:
            val = row.get(h, "")
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = ""
            row_data.append(str(val))
        rows.append(row_data)

    ws.append_rows(rows, value_input_option=ValueInputOption.raw)
    return len(rows)


def get_reviewed_rows() -> pd.DataFrame:
    """
    Read all rows from the 'Review' sheet where review_status == 'reviewed'.

    Returns a pandas DataFrame containing only those rows.
    Returns an empty DataFrame if the sheet is empty or has no reviewed rows.
    """
    ws = get_sheet(REVIEW_SHEET_NAME)
    all_values = ws.get_all_values()

    if len(all_values) < 2:
        return pd.DataFrame()

    headers = all_values[0]
    df = pd.DataFrame(all_values[1:], columns=headers).replace("", None)

    if "review_status" not in df.columns:
        return pd.DataFrame()

    mask = df["review_status"].fillna("").str.strip().str.lower() == "reviewed"
    return df[mask].reset_index(drop=True)