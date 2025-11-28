import collections
from datetime import datetime

from data_pipeline.database.sheets_interface import list_incidents


def main():
    incidents = list_incidents()
    print(f"Total incidents in sheet: {len(incidents)}")

    by_year = collections.Counter()
    by_perpetrator = collections.Counter()
    raid_count = 0
    harm_to_property_count = 0
    dispossession_count = 0

    for inc in incidents:
        # incident_date is stored as "YYYY-MM-DD"
        date_str = inc.get("incident_date") or ""
        try:
            year = datetime.strptime(date_str, "%Y-%m-%d").year
        except ValueError:
            year = "UNKNOWN"

        by_year[year] += 1

        perp = (inc.get("perpetrator") or "").strip() or "UNKNOWN"
        by_perpetrator[perp] += 1

        # booleans are stored as "TRUE"/"FALSE" or empty
        if (inc.get("raid") or "").upper() == "TRUE":
            raid_count += 1
        if (inc.get("harm_to_property") or "").strip():
            harm_to_property_count += 1
        if (inc.get("dispossession") or "").strip():
            dispossession_count += 1

    print("\nIncidents by year:")
    for year, count in sorted(by_year.items(), key=lambda x: str(x[0])):
        print(f"  {year}: {count}")

    print("\nIncidents by perpetrator:")
    for perp, count in by_perpetrator.most_common():
        print(f"  {perp}: {count}")

    print("\nKey categories:")
    print(f"  Incidents coded as raid: {raid_count}")
    print(f"  Incidents with harm_to_property: {harm_to_property_count}")
    print(f"  Incidents with dispossession: {dispossession_count}")


if __name__ == "__main__":
    main()
