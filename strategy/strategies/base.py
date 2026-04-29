"""
strategy/strategies/base.py
----------------
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import date

from strategy.models import Signal, Direction


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

    def compute_price_levels(self, df: pd.DataFrame, *, as_intervals: bool = False, num_points: int = 500,
                             price_range: tuple[float, float] = (0.90, 1.10)) -> tuple[date | None, list]:
        """
        Sweep a range of hypothetical closing prices and return the signal the strategy
        would emit at each price point.
        """

        df = df.copy().sort_index()

        if df.empty:
            return None, []

        latest = df["close"].iloc[-1]
        latest_date = df.index[-1]
        lo, hi = price_range

        prices = np.linspace(latest * lo, latest * hi, num_points)
        results = []

        for p in prices:
            test_df = df.copy()
            test_df.loc[test_df.index[-1], "close"] = p
            results.append((p, self.run(test_df)))

        if as_intervals:
            return latest_date, self._as_intervals(results)

        return latest_date, results

    @staticmethod
    def _as_intervals(data: list[tuple[float, Signal]]) -> list[tuple[float, float, Direction, str | None]]:
        """Convert a sorted ``(price,Signal)`` sequence into direction intervals."""

        if not data:
            return []

        intervals: list[tuple[float, float, Direction, str | None]] = []
        p_start = data[0][0]
        d_current = data[0][1].direction

        for i in range(1, len(data)):
            d_next = data[i][1].direction

            if d_next != d_current:
                reason = data[i - 1][1].metadata.get("strongest_reason")
                intervals.append((p_start, data[i - 1][0], d_current, reason))

                p_start = data[i][0]
                d_current = d_next

        reason = data[-1][1].metadata.get("strongest_reason")
        intervals.append((p_start, data[-1][0], d_current, reason))

        return intervals
