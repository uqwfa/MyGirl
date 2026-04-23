"""
strategy/strategies/base.py
----------------
"""

import pandas as pd
from abc import ABC, abstractmethod

from strategy.models import Signal


class BaseStrategy(ABC):
    """Base abstract strategy class."""

    def __init__(self, params: dict | None = None):
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

    @staticmethod
    def _as_intervals(data: list[tuple[float, Signal]]):
        """get a list of (price, signal) tuples and convert to intervals of (start_price, end_price, signal)"""

        intervals = []
        p_start = data[0][0]

        for i in range(len(data) - 1):
            d_now = data[i][1].direction
            d_next = data[i + 1][1].direction

            if d_now != d_next:
                p_end = data[i][0]
                intervals.append((p_start, p_end, d_now))

                p_start = data[i + 1][0]

        if p_start is not None:
            intervals.append((p_start, data[-1][0], data[-1][1].direction))

        return intervals
