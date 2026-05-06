"""
strategy/strategies/base.py
---------------------------
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import date

from strategy.models import Signal, Direction


class BaseStrategy(ABC):
    """Base abstract strategy class."""

    def __init__(self, params: dict | None = None):
        """Initialize the strategy with optional parameters."""

        self.params = params or {}

    @abstractmethod
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add static indicator columns to the whole DataFrame ``df`` that doesn't need extra information and returns it.
        """

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, *, buy_date: date | None = None) -> Signal:
        """
        Given a ``df`` with indicator columns already attached (e.x. from the ``compute_indicators`` method) and
        additional information, this method returns a ``Signal`` object with the strategy's recommendation for the
        latest date in the DataFrame. The ``buy_date`` parameter is optional and can be used to provide context for
        sell signals, such as the date of purchase.
        """

    def run(self, df: pd.DataFrame, *, buy_date: date | None = None) -> Signal:
        """
        The single public entry point - for backtesting and live trading.

        It enriches the input DataFrame ``df`` with the own ``compute_indicators`` method and returns the recommendation
        ``Signal`` for the latest date in the DataFrame using the own ``generate_signal`` method. The ``buy_date`` parameter
        is optional and can be used to provide context for the signals.
        """

        enriched_df = self.compute_indicators(df.copy())
        return self.generate_signal(enriched_df, buy_date=buy_date)

    def compute_price_levels(self, df: pd.DataFrame, *, as_intervals: bool = False, num_points: int = 500,
                             price_range: tuple[float, float] = (0.90, 1.10), buy_date: date | None = None) -> tuple[date | None, list]:
        """
        Sweep a range of hypothetical closing prices and return the signal the strategy would emit at each price point.

        The price levels are computed for the lastest date in the DataFrame. It works by setting the price
        value of the latest date to the hypothetical closing prices and calling the own ``run`` method. It uses ``run``
        and not ``generate_signal`` because with changing prices also comes changing indicators. The ``buy_date``
        parameter is optional and can be used to provide context for the signals.
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
            results.append((p, self.run(test_df, buy_date=buy_date)))

        if as_intervals:
            return latest_date, self._as_intervals(results)

        return latest_date, results

    @staticmethod
    def _as_intervals(data: list[tuple[float, Signal]]) -> list[tuple[float, float, Direction, str | None]]:
        """
        Convert a sorted ``(price,Signal)`` tuple sequence into direction intervals sequence with the layout:
        ``(start_price,end_price,direction,reason)``.
        """

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
