"""
ingestion/scheduler.py
----------------------
Entry point for updating security prices in the local database.
"""

from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass

from ingestion.collectors.arivaScraper import ArivaScraper
from ingestion.collectors.baseCollector import Collector
from ingestion.normalizer import normalize
from storage.models import Security, DateRange
from storage.repository import add_ohlcv_rows


@dataclass
class UpdateResult:
    """Outcome of a single security update attempt."""

    security: Security
    collector: str
    rows_stored: int = 0
    error: Exception | None = None

    @property
    def success(self) -> bool:
        return self.error is None


_COLLECTOR_REGISTRY: dict[str, Collector] = {
    "ArivaScraper": ArivaScraper()
}


def get_collector(name: str) -> Collector:
    """Return a registered Collector instance by name."""

    collector = _COLLECTOR_REGISTRY.get(name)

    if collector is None:
        registered = ", ".join(_COLLECTOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown collector '{name}'. Registered: {registered}"
        )

    return collector


def schedule_updates(tasks: list[tuple[Security, DateRange, str]], max_workers: int = 4) -> list[UpdateResult]:
    """Run update_security in parallel for every (security, date_range, collector) tuple in *tasks*."""

    if not tasks:
        print("No tasks to schedule.")
        return []

    results: list[UpdateResult] = []
    future_to_security: dict[Future, Security] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for security, date_range, collector_name in tasks:
            future = executor.submit(update_security, security, date_range, collector_name)
            future_to_security[future] = security

        for future in as_completed(future_to_security):
            result: UpdateResult = future.result()
            results.append(result)

            if result.success:
                print(f"[{result.security.isin}] ✓ {result.rows_stored} rows stored.")

            else:
                print(f"[{result.security.isin}] ✗ {result.error}")

    success_count = sum(1 for r in results if r.success)
    total_rows = sum(r.rows_stored for r in results)
    print(f"Completed {success_count}/{len(tasks)} securities - {total_rows} total rows stored.")

    return results


def update_security(security: Security, date_range: DateRange, collector_name: str = "ArivaScraper") -> UpdateResult:
    """Fetch, normalize, and store OHLCV data for a single security."""

    result = UpdateResult(security=security, collector=collector_name)

    try:
        collector = get_collector(collector_name)

        print(f"[{security.isin}] Fetching via {collector_name} for {date_range.start} → {date_range.end}")

        df = collector.fetch(isin=security.isin, date_range=date_range, ariva_id=security.ariva_id)

        rows = normalize(df, isin=security.isin, collector=collector_name)

        if not rows:
            print(f"[{security.isin}] No rows to store after normalization.")
            return result

        result.rows_stored = add_ohlcv_rows(rows)
        print(f"[{security.isin}] Stored {result.rows_stored} rows.")

    except Exception as exc:
        print(f"[{security.isin}] Update failed: {exc}")
        result.error = exc

    return result
