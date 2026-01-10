"""
Simplified date parsing utilities (date-only, no timezone).
"""

from datetime import datetime, date
from typing import Optional
import re


def parse_date_string(date_str: Optional[str]) -> Optional[date]:
    """
    Parse a date string into a date object (no time).
    Handles incomplete dates like "October 20" by assuming current year.
    """
    if not date_str:
        return None
    
    date_str = date_str.strip()
    current_year = date.today().year
    
    # Try patterns with year first
    patterns = [
        # Full dates with year
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: date(int(m[3]), int(m[2]), int(m[1]))),
        (r'([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', 
         lambda m: date(int(m[3]), _month_to_number(m[1]), int(m[2]))),
        
        # Dates without year - add current year
        (r'([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?', 
         lambda m: date(current_year, _month_to_number(m[1]), int(m[2]))),
        (r'(\d{1,2})\s+([A-Za-z]+)', 
         lambda m: date(current_year, _month_to_number(m[2]), int(m[1]))),
    ]
    
    # ... rest of function ...
    
    for pattern, converter in patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                return converter(match.groups())
            except (ValueError, IndexError):
                continue
    
    # Try using dateutil as fallback (but keep it simple)
    try:
        import dateutil.parser
        dt = dateutil.parser.parse(date_str, fuzzy=True)
        return dt.date()
    except:
        pass
    
    return None


def _month_to_number(month_str: str) -> int:
    """Convert month name to number (1-12)."""
    months = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }
    return months.get(month_str.lower().strip(), 1)


def format_date_for_sheets(dt: date) -> str:
    """
    Format date for Google Sheets (ISO 8601 format).
    
    Returns:
        ISO 8601 date string (YYYY-MM-DD)
    """
    return dt.isoformat()


def is_valid_date(year: int, month: int, day: int) -> bool:
    """
    Check if a date is valid.
    
    Returns:
        True if valid date
    """
    try:
        date(year, month, day)
        return True
    except ValueError:
        return False