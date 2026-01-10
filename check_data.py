#!/usr/bin/env python3
"""
Check sample data from Google Sheets.
"""
import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, Any, List

creds = Credentials.from_service_account_file('service_account.json', scopes=['https://www.googleapis.com/auth/spreadsheets'])
client = gspread.authorize(creds)
sh = client.open_by_key('1Z7zu2JLxOIU1yK3SrXIz8yN3a-2Kzd7d0g0-VNrkwaI')
ws = sh.worksheet('Reports')

print('=== SAMPLE OF DATA ===')
data: List[Dict[str, Any]] = ws.get_all_records()[:3]  # First 3 records as dicts

for i, record in enumerate(data, 1):
    print(f'\n--- Report R{str(i).zfill(5)} ---')
    
    # Safely get title with type checking
    title = record.get("title", "")
    if isinstance(title, str):
        print(f'Title: {title[:80]}...')
    else:
        print(f'Title: (not a string: {type(title)})')
    
    # Safely get other fields
    date_val = record.get("published_date", "")
    print(f'Date: {date_val if isinstance(date_val, str) else ""}')
    
    location = record.get("location_raw", "")
    print(f'Location: {location if isinstance(location, str) else ""}')
    
    actors = record.get("actors_raw", "")
    print(f'Actors: {actors if isinstance(actors, str) else ""}')
    
    url = record.get("url", "")
    print(f'URL: {url if isinstance(url, str) else ""}')