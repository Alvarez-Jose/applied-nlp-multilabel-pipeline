"""
Diagnostic script — prints the real structure of the Google Sheets spreadsheet
so we can verify the pipeline integration is correct.

Run with:
    python check_sheets_structure.py
"""

import json
import sys
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

SPREADSHEET_ID     = "1Z7zu2JLxOIU1yK3SrXIz8yN3a-2Kzd7d0g0-VNrkwaI"
SERVICE_ACCOUNT    = Path(__file__).parent / "service_account.json"
SCOPES             = ["https://www.googleapis.com/auth/spreadsheets"]
SAMPLE_ROWS        = 3   # how many data rows to preview per sheet


def connect():
    if not SERVICE_ACCOUNT.exists():
        sys.exit(f"ERROR: service_account.json not found at {SERVICE_ACCOUNT}")
    creds  = Credentials.from_service_account_file(str(SERVICE_ACCOUNT), scopes=SCOPES)
    client = gspread.authorize(creds)
    return client.open_by_key(SPREADSHEET_ID)


def inspect_sheet(ws):
    all_vals = ws.get_all_values()
    if not all_vals:
        print("    (empty — no rows at all)")
        return

    headers   = all_vals[0]
    data_rows = all_vals[1:]

    print(f"    Rows (excl. header): {len(data_rows)}")
    print(f"    Columns ({len(headers)}): {headers}")

    if data_rows:
        print(f"    --- First {min(SAMPLE_ROWS, len(data_rows))} data row(s) ---")
        for i, row in enumerate(data_rows[:SAMPLE_ROWS]):
            row_dict = dict(zip(headers, row))
            # Truncate long text fields for readability
            trimmed = {k: (v[:80] + "...") if len(str(v)) > 80 else v
                       for k, v in row_dict.items()}
            print(f"    [{i+1}] {json.dumps(trimmed, ensure_ascii=False, indent=6)}")


def main():
    print(f"Connecting to spreadsheet: {SPREADSHEET_ID}\n")
    try:
        sh = connect()
    except Exception as e:
        sys.exit(f"ERROR connecting: {e}")

    worksheets = sh.worksheets()
    print(f"Tabs found ({len(worksheets)}): {[ws.title for ws in worksheets]}\n")

    for ws in worksheets:
        print(f"=== Tab: '{ws.title}' ===")
        try:
            inspect_sheet(ws)
        except Exception as e:
            print(f"    ERROR reading tab: {e}")
        print()


if __name__ == "__main__":
    main()
