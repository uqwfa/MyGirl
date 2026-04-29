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


class BookStrategy(BaseStrategy):
    """
    An implementation of the BaseStrategy class.

    Params (all optional, with defaults):
        bb_window      (int. default 20)
        bb_factor      (float, default 2.0)
        sell_factor    (float, default 0.96)
        drawdown_limit (float, default 0.80)
    """

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()

        bb_window = self.params.get("bb_window", 20)
        bb_factor = self.params.get("bb_factor", 2.0)
        df = add_bb(df, bb_window, bb_factor)

        df["ma_4"] = df["close"].rolling(4).mean()
        df["ma_9"] = df["close"].rolling(9).mean()
        df["ma_18"] = df["close"].rolling(18).mean()

        return df

    def generate_signal(self, df: pd.DataFrame) -> Signal:
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
            last = df.iloc[-1]
            return self._score(
                price=last["close"],
                current_date=df.index[-1],
                bb_upper=last["bb_upper"],
                bb_lower=last["bb_lower"],
                ma_4=last["ma_4"],
                ma_9=last["ma_9"],
                ma_18=last["ma_18"],
                period_high=df["close"].max(),
                strategy_name=strategy_name
            )

        except Exception as exc:
            print(f"[{strategy_name}] Error generating signal: {exc}")

            return Signal(
                direction=Direction.INVALID,
                strength=None,
                date=None,
                strategy=strategy_name,
                metadata={"error": str(exc)}
            )

    def compute_price_levels(self, df: pd.DataFrame, *, as_intervals: bool = False, num_points: int = 500,
                             price_range: tuple[float, float] = (0.80, 1.20)) -> tuple[date | None, list]:
        df = df.copy().sort_index()

        if df.empty:
            return None, []

        latest = float(df["close"].iloc[-1])
        latest_date = df.index[-1]
        close = df["close"].values
        strategy_name = self.__class__.__name__

        bb_window = self.params.get("bb_window", 20)
        bb_factor = self.params.get("bb_factor", 2.0)


        min_rows = max(bb_window, 18)  # 18 = longest MA
        if len(df) < min_rows:
            raise ValueError(f"DataFrame has {len(df)} rows but at least {min_rows} are required to compute all indicators.")

        # precompute historical context (everything except last row)
        hist_close = close[:-1]

        sum_4 = hist_close[-3:].sum()  # last 3 real closes  → + p = MA4
        sum_9 = hist_close[-8:].sum()  # last 8 real closes  → + p = MA9
        sum_18 = hist_close[-17:].sum()  # last 17 real closes → + p = MA18

        hist_bb = hist_close[-(bb_window - 1):]
        period_high = float(hist_close.max())

        lo, hi = price_range
        prices = np.linspace(latest * lo, latest * hi, num_points)

        ma_4 = (sum_4 + prices) / 4
        ma_9 = (sum_9 + prices) / 9
        ma_18 = (sum_18 + prices) / 18

        bb_upper, bb_lower = bb_at_price(hist_bb, prices, bb_window, bb_factor)

        results = [
            (
                float(p),
                self._score(
                    price=float(p),
                    current_date=latest_date,
                    bb_upper=float(bb_upper[i]),
                    bb_lower=float(bb_lower[i]),
                    ma_4=float(ma_4[i]),
                    ma_9=float(ma_9[i]),
                    ma_18=float(ma_18[i]),
                    period_high=period_high,
                    strategy_name=strategy_name
                )
            )
            for i, p in enumerate(prices)
        ]

        if as_intervals:
            return latest_date, self._as_intervals(results)

        return latest_date, results

    def _score(self, *, price: float, current_date : date, bb_upper: float, bb_lower: float, ma_4: float,
               ma_9 : float, ma_18: float, period_high: float, strategy_name: str) -> Signal:
        """Evaluate all conditions for explicitly provided indicator values and return the strongest :class:`Signal`."""

        long_candidates: list[tuple[float, str]] = []
        short_candidates: list[tuple[float, str]] = []

        # long conditions
        if price < bb_lower:
            strength = 1.0 - (price / bb_lower)
            long_candidates.append((strength, "Below lower BB."))

        if ma_4 > ma_9 > ma_18:
            long_candidates.append((1.0, "MA4 > MA9 > MA18 momentum alignment."))

        # short conditions
        if price > bb_upper:
            strength = (price / bb_upper) - 1.0
            short_candidates.append((strength, "Above upper BB."))

        sell_factor = self.params.get("sell_factor", 0.96)
        ma_short_threshold = sell_factor * ma_18
        if ma_4 < ma_short_threshold:
            strength = 1.0 - (ma_4 / ma_short_threshold)
            short_candidates.append((strength, "MA4 below sell_factor × MA18."))

        drawdown_limit = self.params.get("drawdown_limit", 0.80)
        drawdown_threshold = drawdown_limit * period_high
        if price < drawdown_threshold:
            strength = 1.0 - (price / drawdown_threshold)
            short_candidates.append((strength, "Close below drawdown limit."))

        if long_candidates:
            best_strength, best_reason = max(long_candidates, key=lambda x: x[0])

            return Signal(
                direction=Direction.LONG,
                strength=round(min(best_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={"reasons": [r for _, r in long_candidates], "strongest_reason": best_reason}
            )

        if short_candidates:
            best_strength, best_reason = max(short_candidates, key=lambda x: x[0])

            return Signal(
                direction=Direction.SHORT,
                strength=round(min(best_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={"reasons": [r for _, r in short_candidates], "strongest_reason": best_reason}
            )

        return Signal(
            direction=Direction.FLAT,
            strength=None,
            date=current_date,
            strategy=strategy_name,
            metadata={}
        )
