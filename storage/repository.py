"""
storage/repository.py
---------------------
Database data access layer.
"""

import pandas as pd

from storage.database import get_connection
from storage.models import OHLCVRow, Security, DateRange


def upsert_security(security: Security) -> None:
    """Insert a security, or update its name and ariva_id if it already exists."""

    sql = """
        INSERT INTO securities (isin, name, ariva_id)
        VALUES (?, ?, ?)
        ON CONFLICT (isin) DO UPDATE SET
            name = excluded.name,
            ariva_id = excluded.ariva_id
    """

    with get_connection() as conn:
        conn.execute(sql, (security.isin, security.name, security.ariva_id))


def add_ohlcv_rows(rows: list[OHLCVRow]) -> int:
    """
    Insert a list of OHLCV rows into the database. Returns the number of rows inserted.
    """

    if not rows:
        return 0

    sql = """
        INSERT INTO ohlcv (security_isin, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (security_isin, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume
    """

    records = [
        (
            r.isin,
            r.date.isoformat(),
            r.open,
            r.high,
            r.low,
            r.close,
            r.volume,
        ) for r in rows
    ]

    with get_connection() as conn:
        conn.executemany(sql, records)

    return len(rows)


def fetch_ohlcv(isin: str, date_range: DateRange) -> pd.DataFrame:
    """Fetch OHLCV data for a security over a date range."""

    sql = """
        SELECT date, open, high, low, close, volume
        FROM ohlcv WHERE security_isin = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """

    params = (isin, date_range.start.isoformat(), date_range.end.isoformat())

    with get_connection() as conn:
        df = pd.read_sql(sql, conn, index_col="date", params=params)

    df.index = pd.to_datetime(df.index)

    return df
