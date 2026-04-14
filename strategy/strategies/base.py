"""
strategy/strategies/base.py
----------------
"""

import pandas as pd
from abc import ABC, abstractmethod

from strategy.models import Signal


class BaseStrategy(ABC):
    """Base abstract strategy class."""

    def __init__(self, params: dict = None):
        self.params = params or {}

    @abstractmethod
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to df and return it."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Given a df with indicator columns already attached, return a Signal object."""

    def run(self, df: pd.DataFrame) -> Signal:
        """The single public entry point - for backtesting and live trading."""

        enriched_df = self.compute_indicators(df.copy())
        return self.generate_signal(enriched_df)
