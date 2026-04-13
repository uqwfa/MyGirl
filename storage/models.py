"""
storage/models.py
-----------------
Standardized models for the storage layer.
"""

from dataclasses import dataclass
from datetime import date


@dataclass
class Security:
    """A security as stored in the database."""

    isin: str
    name: str
    ariva_id: int


@dataclass
class OHLCVRow:
    """A single daily OHLCV row for one security."""

    isin: str
    date: date
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: int | None = None


@dataclass
class DateRange:
    """An inclusive date range."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must not be after end ({self.end})")

    def __contains__(self, item: date) -> bool:
        return self.start <= item <= self.end
    