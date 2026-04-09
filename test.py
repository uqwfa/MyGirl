from ingestion.collectors.arivaScraper import ArivaScraper
from storage.models import DateRange
from datetime import date


d = DateRange(start=date(1990, 1, 1), end=date(1990, 1, 3))
print(ArivaScraper().fetch("DE0007164600", d, 6))
