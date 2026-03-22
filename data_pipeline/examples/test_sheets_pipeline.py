from datetime import date

from data_pipeline.database.models import Report
from data_pipeline.database.sheets_interface import create_incident, add_reports


def main():
    # 1. Create an incident
    incident_date = date(2025, 3, 12)
    incident_id = create_incident(
        governorate_code="HEB",
        incident_date=incident_date,
        location_norm="Hebron, Old City",
        incident_type="raid",
    )
    print("Created incident:", incident_id)

    # 2. Create one fake report (date-only -- no timezone handling needed)
    report = Report(
        source_name="Example News",
        url="https://example.com/article1",
        title="Soldiers raid home in Hebron",
        body="Full article text...",
        published_date=date(2025, 3, 12),
        language="en",
        location_raw="Hebron",
        actors_raw="Israeli army; Palestinian residents",
    )

    add_reports([report])
    print("Inserted 1 report into Google Sheets.")


if __name__ == "__main__":
    main()
