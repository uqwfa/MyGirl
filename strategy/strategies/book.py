"""
strategy/strategies/book.py
---------------------------
"""

import pandas as pd

from strategy.strategies.base import BaseStrategy
from strategy.models import Signal, Direction
from strategy.indicators.bollingerBands import add_bb


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
            return self._evaluate(df, strategy_name)

        except Exception as exc:
            print(f"Error generating signal in {strategy_name}")

            return Signal(
                direction=Direction.INVALID,
                strength=None,
                date=None,
                strategy=strategy_name,
                metadata={"error": str(exc)}
            )

    def _evaluate(self, df: pd.DataFrame, strategy_name: str) -> Signal:
        """Core signal logic. Evaluates all conditions and returns the strongest signal."""

        latest = df["close"].iloc[-1]
        current_date = df.index[-1]

        bb_upper = df["bb_upper"].iloc[-1]
        bb_lower = df["bb_lower"].iloc[-1]
        ma_4 = df["ma_4"].iloc[-1]
        ma_9 = df["ma_9"].iloc[-1]
        ma_18 = df["ma_18"].iloc[-1]

        long_candidates: list[tuple[float, str]] = []

        if latest < bb_lower:
            strength = 1.0 - (latest / bb_lower)
            long_candidates.append((strength, "Below lower BB."))

        if ma_4 > ma_9 > ma_18:
            long_candidates.append((1.0, "MA4 > MA9 > MA18 momentum alignment."))

        short_candidates: list[tuple[float, str]] = []

        if latest > bb_upper:
            strength = (latest / bb_upper) - 1.0
            short_candidates.append((strength, "Above upper BB."))

        ma_short_threshold = self.params.get("sell_factor", 0.96) * ma_18
        if ma_4 < ma_short_threshold:
            strength = 1.0 - (ma_4 / ma_short_threshold)
            short_candidates.append((strength, "MA4 below sell_factor * MA18."))

        period_high = df["close"].max()
        drawdown_threshold = self.params.get("drawdown_limit", 0.80) * period_high
        if latest < drawdown_threshold:
            strength = 1.0 - (latest / drawdown_threshold)
            short_candidates.append((strength, "Close below drawdown limit."))

        if long_candidates:
            best_strength, best_reason = max(long_candidates, key=lambda x: x[0])
            all_reasons = [r for _, r in long_candidates]

            return Signal(
                direction=Direction.LONG,
                strength=round(min(best_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={"reasons": all_reasons, "strongest_reason": best_reason}
            )

        if short_candidates:
            best_strength, best_reason = max(short_candidates, key=lambda x: x[0])
            all_reasons = [r for _, r in short_candidates]

            return Signal(
                direction=Direction.SHORT,
                strength=round(min(best_strength, 1.0), 6),
                date=current_date,
                strategy=strategy_name,
                metadata={"reasons": all_reasons, "strongest_reason": best_reason},
            )

        return Signal(
            direction=Direction.FLAT,
            strength=None,
            date=current_date,
            strategy=strategy_name,
            metadata={},
        )
