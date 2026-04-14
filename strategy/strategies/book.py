"""
strategy/strategies/book.py
---------------------------
"""

import pandas as pd

from strategy.strategies.base import BaseStrategy
from strategy.models import Signal, Direction
from strategy.indicators.bollingerBands import add_bb


class BookStrategy(BaseStrategy):
    """An implementation of the BaseStrategy class."""

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()

        bb_window = self.params.get("bb_window", 20)
        bb_factor = self.params.get("bb_factor", 2)
        df = add_bb(df, bb_window, bb_factor)

        df["ma_4"] = df["close"].rolling(4).mean()
        df["ma_9"] = df["close"].rolling(9).mean()
        df["ma_18"] = df["close"].rolling(18).mean()

        return df

    def generate_signal(self, df: pd.DataFrame) -> Signal | None:
        if df.empty:
            return Signal(
                direction=Direction.NAN,
                strength=-1,
                date=None,
                strategy=self.__class__.__name__,
                metadata={"error": "Empty DataFrame."}
            )

        try:
            latest_value = df["close"].sort_index().iloc[-1]
            max_value = df["close"].max()
            current_date = df.index[-1]

            direction_reason = []
            direction = Direction.FLAT
            strength = 0.0

            if latest_value < df["bb_lower"].iloc[-1]:
                direction = Direction.LONG
                strength = 1 - (latest_value / df["bb_lower"].iloc[-1])
                direction_reason.append("Below bb bands.")

            if (df["ma_4"].iloc[-1] > df["ma_9"].iloc[-1]) and (df["ma_9"].iloc[-1] > df["ma_18"].iloc[-1]):
                direction = Direction.LONG
                strength = 1.0
                direction_reason.append("Ma4 > Ma9 > Ma18.")

            if direction == Direction.LONG:
                return Signal(
                    direction=direction,
                    strength=strength,
                    date=current_date,
                    strategy=self.__class__.__name__,
                    metadata={"direction_reason": direction_reason}
                )

            if latest_value > df["bb_upper"].iloc[-1]:
                direction = Direction.SHORT
                strength = (latest_value / df["bb_upper"].iloc[-1]) - 1
                direction_reason.append("Above bb bands.")

            sell_factor = self.params.get("sell_factor", 0.96)
            ma_short_threshold = sell_factor * df["ma_18"].iloc[-1]

            if df["ma_4"].iloc[-1] < ma_short_threshold:
                direction = Direction.SHORT
                strength = 1 - (df["ma_4"].iloc[-1] / ma_short_threshold)
                direction_reason.append("Ma4 < sell_factor * Ma18.")

            drawdown_limit = self.params.get("drawdown_limit", 0.8)
            drawdown_threshold = drawdown_limit * max_value

            if latest_value < drawdown_threshold:
                direction = Direction.SHORT
                strength = 1 - (latest_value / drawdown_threshold)
                direction_reason.append("Price below drawdown limit.")

            if direction == Direction.SHORT:
                return Signal(
                    direction=direction,
                    strength=strength,
                    date=current_date,
                    strategy=self.__class__.__name__,
                    metadata={"direction_reason": direction_reason}
                )

            return Signal(
                direction=Direction.FLAT,
                strength=-1,
                date=current_date,
                strategy=self.__class__.__name__,
                metadata={}
            )

        except Exception as e:
            print(f"Error generating signal in {self.__class__.__name__}: {e}")

            return Signal(
                direction=Direction.NAN,
                strength=-1,
                date=None,
                strategy=self.__class__.__name__,
                metadata={"error": str(e)}
            )
