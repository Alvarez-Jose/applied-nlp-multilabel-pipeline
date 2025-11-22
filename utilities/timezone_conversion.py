from datetime import datetime
from zoneinfo import ZoneInfo

PALESTINE_TZ = ZoneInfo("Asia/Hebron")  # or "Asia/Gaza"
UTC_TZ = ZoneInfo("UTC")

def to_palestine_time(dt_utc: datetime) -> datetime:
    return dt_utc.astimezone(PALESTINE_TZ)