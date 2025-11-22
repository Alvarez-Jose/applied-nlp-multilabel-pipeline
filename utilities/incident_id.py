from datetime import date

def format_incident_id(governorate_code: str, incident_data: date, sequence_num: int)-> str:
    '''Keeps the ID concept from the code'''
    date_str = incident_data.strftime("%Y%m%d")
    seq_str = f"{sequence_num:02d}"  # 1 -> "01"
    return f"{governorate_code}-{date_str}-{seq_str}"