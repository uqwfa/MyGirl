from datetime import date

from ingestion.scheduler import schedule_updates
from storage.database import init_db
from storage.models import Security, DateRange

if __name__ == "__main__":
    init_db()
    tasks = [
        (Security(isin="US6311011026", name="NASDAQ 100 Index", ariva_id=72),
         DateRange(start=date(2026, 4, 2), end=date(2026, 4, 11)),
         "ArivaScraper")
    ]
    schedule_updates(tasks)
