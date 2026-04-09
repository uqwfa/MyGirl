"""
ingestion/collectors/baseCollector.py
-------------------------------------
Base abstract collector class.
"""

from abc import ABC, abstractmethod
from datetime import date


class Collector(ABC):
    @abstractmethod
    def fetch(self, isin: str, start_date: date, end_date: date, **kwargs) -> any:
        """Abstract method to fetch data for a given ISIN and date range."""
