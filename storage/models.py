"""
storage/models.py
-----------------
Standardized models for the storage layer.
"""

from dataclasses import dataclass
from datetime import date


@dataclass
class OHLCVRow:
    """A single daily OHLCV row for one security."""

    isin: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    