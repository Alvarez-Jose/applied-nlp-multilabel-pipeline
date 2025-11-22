from gspread import Client
from google.oauth2.service_account import Credentials

from datetime import date
from typing import List, Dict, Optional

import gspread

from data_pipeline.database.models import Report
# from utilities import incident_id   # <- you don't actually need this
from utilities.incident_id import format_incident_id

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Path to the service account JSON (located in project root)
SERVICE_ACCOUNT_FILE = "service_account.json"

# Spreadsheet ID (long string in the Google Sheet URL)
SPREADSHEET_ID = "1Z7zu2JLxOIU1yK3SrXIz8yN3a-2Kzd7d0g0-VNrkwaI"

INCIDENTS_SHEET_NAME = "Incidents"
REPORTS_SHEET_NAME = "Reports"


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
    """Return all incidents as a list of dicts. Assumes row 1 is header in the 'Incidents' sheet."""
    ws = get_sheet(INCIDENTS_SHEET_NAME)
    rows = ws.get_all_records()  # list[dict] with keys from header row
    return rows


def get_next_sequence(governorate_code: str, incident_date: date) -> int:
    """Count how many incidents already exist for this governorate + date, then assign the next sequence number."""
    incidents = list_incidents()
    target_date_str = incident_date.isoformat()

    count = sum(
        1
        for inc in incidents
        if inc.get("governorate_code") == governorate_code
        and inc.get("incident_date") == target_date_str
    )

    return count + 1


def create_incident(
    governorate_code: str,
    incident_date: date,
    location_norm: str,
    incident_type: Optional[str] = None,
) -> str:
    """Create a new incident row in the Incidents sheet and return its incident_id."""
    ws = get_sheet(INCIDENTS_SHEET_NAME)

    seq = get_next_sequence(governorate_code, incident_date)
    incident_id = format_incident_id(governorate_code, incident_date, seq)

    # This order MUST match the header row:
    # incident_id | governorate_code | incident_date | location_norm | incident_type | conflict_casualty_count | conflict_perpetrator
    row = [
        incident_id,
        governorate_code,
        incident_date.isoformat(),  # <- FIXED: call isoformat()
        location_norm,
        incident_type or "",
        "FALSE",  # conflict_casualty_count
        "FALSE",  # conflict_perpetrator
    ]

    ws.append_row(row)
    return incident_id


def add_reports(reports: List[Report]) -> None:
    """
    Append a batch of reports to the Reports sheet.

    Header for Reports MUST be:
    report_id | incident_id | source_name | url | title | body_text |
    published_at_utc | published_at_local | language | location_raw | actors_raw
    """
    ws = get_sheet(REPORTS_SHEET_NAME)

    # Get all existing rows to estimate next report_id index
    existing = ws.get_all_values()
    # existing includes header row; number of data rows:
    data_row_count = max(len(existing) - 1, 0)

    rows_to_append = []
    for i, r in enumerate(reports):
        report_number = data_row_count + i + 1  # 1-based
        report_id = f"R{report_number:05d}"     # e.g. R00001

        # Make sure everything is a string
        published_utc_str = r.published_at_utc.isoformat()
        published_local_str = r.published_at_local.isoformat()

        row = [
            report_id,
            "",  # incident_id (future: linking logic)
            r.source_name or "",
            r.url or "",
            r.title or "",
            r.body or "",
            published_utc_str,
            published_local_str,
            r.language or "",
            r.location_raw or "",
            r.actors_raw or "",
        ]
        rows_to_append.append(row)

    # Append all rows in one API call
    ws.append_rows(rows_to_append)

    