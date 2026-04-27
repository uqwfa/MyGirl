from datetime import date, timedelta

from ingestion.scheduler import schedule_updates
from storage.database import init_db
from storage.models import Security, DateRange
from storage.repository import fetch_ohlcv
from strategy.strategies.book import BookStrategy


if __name__ == "__main__":
    init_db()

    today = date.today()
    d = DateRange(start=(today-timedelta(days=30)), end=today)
    d2 = DateRange(start=date(2025, 4, 28), end=today)

    tasks = [
        (
            Security(isin="US6311011026", name="NASDAQ 100 Index", ariva_id=72),
            d,
            "ArivaScraper"
        ),

        (
            Security(isin="DE0007164600", name="SAP SE", ariva_id=6),
            d,
            "ArivaScraper"
        )
    ]
    schedule_updates(tasks)

    x = fetch_ohlcv("US6311011026", d2)
    b = BookStrategy()

    print(f"\nCurrent signal:\n{b.run(x)}")
    print(f"\nPrice Levels:\n{b.compute_price_levels(x, as_intervals=True)}")
