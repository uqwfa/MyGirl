"""
ingestion/collectors/arivaScraper.py
------------------------------------
"""

import pandas as pd
import requests
from datetime import date

from ingestion.collectors.baseCollector import Collector


class ArivaScraper(Collector):

    def fetch(self, isin: str, start_date: date, end_date: date, ariva_id: int = None, **kwargs) -> pd.DataFrame:
        if ariva_id is None:
            print(f"Ariva ID not provided for {isin}")
            return pd.DataFrame()

        # todo: implement logic
        return pd.DataFrame()
