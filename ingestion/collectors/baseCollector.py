"""
ingestion/collectors/baseCollector.py
-------------------------------------
"""

import pandas as pd
from abc import ABC, abstractmethod

from storage.models import DateRange


class Collector(ABC):
    """Base abstract collector class."""

    @abstractmethod
    def fetch(self, isin: str, date_range: DateRange, **kwargs) -> pd.DataFrame:
        """Abstract method to fetch data for a given ISIN and date range."""
