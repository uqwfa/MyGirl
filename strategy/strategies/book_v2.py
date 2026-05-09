"""
strategy/strategies/book.py
---------------------------
"""

import numpy as np
import pandas as pd
from datetime import date

from strategy.strategies.base import BaseStrategy
from strategy.models import Signal, Direction
from strategy.indicators.bollingerBands import add_bb, bb_at_price


class BookStrategyV2(BaseStrategy):
    """
    An implementation of the BaseStrategy class. This strategy uses Moving Averages (MA) and Bollinger Bands (BB).

    Params (all optional, with defaults):
        bb_window      (int. default 20)
        bb_factor      (float, default 2.0)
        ma_short       (short moving average)  # later be named to ma_short
        ma_medium      (medium moving average) # later be named to ma_medium
        ma_long        (long moving average)   # later be named to ma_long
        sell_factor    (float, default 0.96)
        drawdown_limit (float, default 0.80)
    """

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()

        bb_window = self.params.get("bb_window", 20)
        bb_factor = self.params.get("bb_factor", 2.0)
        df = add_bb(df, bb_window, bb_factor)

        ma_short = int(self.params.get("ma_short", 4))
        ma_medium = int(self.params.get("ma_medium", 9))
        ma_long = int(self.params.get("ma_long", 18))
        df["ma_short"] = df["close"].rolling(ma_short).mean()
        df["ma_medium"] = df["close"].rolling(ma_medium).mean()
        df["ma_long"] = df["close"].rolling(ma_long).mean()

        return df

    def generate_signal(self, df: pd.DataFrame, *, buy_date: date = None) -> Signal:
        strategy_name = self.__class__.__name__

        if df.empty:
            return Signal(
                direction=Direction.INVALID,
                strength=None,
                date=None,
                strategy=strategy_name,
                metadata={"error": "Empty DataFrame."}
            )

        try:
            # Since this strategy needs the period maximum value for the sell signal, starting from the buy_date till
            # the current last date of the DataFrame, we need to find it. Set it to 0.0 if the buy_date is None, so
            # that the period_high value doesn't change the result.
            last = df.iloc[-1]
            period_high = 0.0
            if buy_date is not None:
                period_high = df.loc[buy_date:, "close"].max()

            return self._score(
                price=last["close"],
                current_date=df.index[-1],
                bb_upper=last["bb_upper"],
                bb_lower=last["bb_lower"],
                ma_short=last["ma_short"],
                ma_medium=last["ma_medium"],
                ma_long=last["ma_long"],
                period_high=period_high,
                strategy_name=strategy_name
            )

        except Exception as exc:
            # print(f"[{strategy_name}] Error generating signal: {exc}")

            return Signal(
                direction=Direction.INVALID,
                strength=None,
                date=None,
                strategy=strategy_name,
                metadata={"error": str(exc)}
            )

    def compute_price_levels(self, df: pd.DataFrame, *, as_intervals: bool = False, num_points: int = 500,
                             price_range: tuple[float, float] = (0.90, 1.10),
                             buy_date: date | None = None) -> tuple[date | None, list]:
        """
        Own implementation of the ``compute_price_levels`` method, because severe optimizations can be made for this
        strategy. For a more general description, see the base class method.

        The method already precomputes parts of different indicators.
        """
        df = df.copy().sort_index()

        if df.empty:
            return None, []

        latest = float(df["close"].iloc[-1])
        latest_date = df.index[-1]
        close = df["close"].values
        strategy_name = self.__class__.__name__

        bb_window = int(self.params.get("bb_window", 20))
        bb_factor = float(self.params.get("bb_factor", 2.0))
        ma_short = int(self.params.get("ma_short", 4))
        ma_medium = int(self.params.get("ma_medium", 9))
        ma_long = int(self.params.get("ma_long", 18))

        min_rows = max(bb_window, ma_long)
        if len(df) < min_rows:
            raise ValueError(f"DataFrame has {len(df)} rows but at least {min_rows} are required to compute all indicators.")

        # precompute historical context (everything except last row)
        hist_close = close[:-1]

        sum_short = hist_close[-(ma_short - 1):].sum()  # last (ma_short - 1) real closes  → + p = MA_SHORT
        sum_medium = hist_close[-(ma_medium - 1):].sum()
        sum_long = hist_close[-(ma_long - 1):].sum()

        hist_bb = hist_close[-(bb_window - 1):]

        # Since this strategy needs the period maximum value for the sell signal, starting from the buy_date till
        # the current last date of the DataFrame, we need to find it. Set it to 0.0 if the buy_date is None, so
        # that the period_high value doesn't change the result.
        period_high = 0.0
        if buy_date is not None:
            period_high = df.loc[buy_date:, "close"].max()

        lo, hi = price_range
        prices = np.linspace(latest * lo, latest * hi, num_points)

        ma_short_value = (sum_short + prices) / ma_short
        ma_medium_value = (sum_medium + prices) / ma_medium
        ma_long_value = (sum_long + prices) / ma_long

        bb_upper, bb_lower = bb_at_price(hist_bb, prices, bb_window, bb_factor)

        results = [
            (
                float(p),
                self._score(
                    price=float(p),
                    current_date=latest_date,
                    bb_upper=float(bb_upper[i]),
                    bb_lower=float(bb_lower[i]),
                    ma_short=float(ma_short_value[i]),
                    ma_medium=float(ma_medium_value[i]),
                    ma_long=float(ma_long_value[i]),
                    period_high=period_high,
                    strategy_name=strategy_name
                )
            )
            for i, p in enumerate(prices)
        ]

        if as_intervals:
            return latest_date, self._as_intervals(results)

        return latest_date, results

    def _score(self, *, price: float, current_date : date, bb_upper: float, bb_lower: float, ma_short: float,
               ma_medium: float, ma_long: float, period_high: float, strategy_name: str) -> Signal:
        """
        Hearth of the strategy. Evaluates all conditions based on all the given values and returns the recommended
        ``Signal``.
        """
        # raise if a single value is nan
        if (pd.isna(price) or pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(ma_short) or pd.isna(ma_medium)
                or pd.isna(ma_long) or pd.isna(period_high)):
            raise ValueError("One or more input values are NaN.")

        long_candidates: list[tuple[float, str]] = []
        short_candidates: list[tuple[float, str]] = []

        # buy (long) conditions
        if price < bb_lower:
            strength = 1.0 - (price / bb_lower)
            long_candidates.append((strength, "Below lower BB."))

        if ma_short > ma_medium > ma_long:
            long_candidates.append((1.0, f"MA_SHORT > MA_MEDIUM > MA_LONG momentum alignment."))

        # sell (short) conditions
        if price > bb_upper:
            strength = (price / bb_upper) - 1.0
            short_candidates.append((strength, "Above upper BB."))

        sell_factor = self.params.get("sell_factor", 0.96)
        ma_short_threshold = sell_factor * ma_long
        if ma_short < ma_short_threshold:
            strength = 1.0 - (ma_short / ma_short_threshold)
            short_candidates.append((strength, "MA_SHORT below sell_factor × MA_LONG."))

        drawdown_limit = self.params.get("drawdown_limit", 0.80)
        drawdown_threshold = drawdown_limit * period_high
        if price < drawdown_threshold:
            strength = 1.0 - (price / drawdown_threshold)
            short_candidates.append((strength, "Close below drawdown limit."))

        best_long = max(long_candidates, key=lambda x: x[0]) if long_candidates else None
        best_short = max(short_candidates, key=lambda x: x[0]) if short_candidates else None

        long_strength = best_long[0] if best_long else 0.0
        short_strength = best_short[0] if best_short else 0.0

        net = long_strength - short_strength

        if net > 0:
            return Signal(
                direction=Direction.LONG,
                strength=round(min(long_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={
                    "reasons": [r for _, r in long_candidates],
                    "strongest_reason": best_long[1],
                    "net_score": round(net, 6),
                },
            )

        if net < 0:
            return Signal(
                direction=Direction.SHORT,
                strength=round(min(short_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={
                    "reasons": [r for _, r in short_candidates],
                    "strongest_reason": best_short[1],
                    "net_score": round(net, 6),
                },
            )

        return Signal(
            direction=Direction.FLAT,
            strength=None,
            date=current_date,
            strategy=strategy_name,
            metadata={"net_score": 0.0},
        )
