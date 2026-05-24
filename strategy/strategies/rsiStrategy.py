from datetime import date

import numpy as np
import optuna
import pandas as pd

from strategy.models import Signal, Direction
from strategy.strategies.base import BaseStrategy


class SimpleRSIStrategy(BaseStrategy):

    def param_space(trial: optuna.Trial):
        return {
            "rsi_period": trial.suggest_int("rsi_period", 10, 18),
            "rsi_up_threshold": trial.suggest_int("rsi_up_threshold", 20, 60),
            "rsi_down_threshold": trial.suggest_int("rsi_down_threshold", 50, 90)
        }

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_index()

        rsi_period = int(self.params.get("rsi_period", 14))
        delta = df["close"].diff()
        gains = delta.clip(lower=0)
        losses = -1 * delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1/rsi_period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))

        return df

    def generate_signal(self, df: pd.DataFrame, *, buy_date: date | None = None) -> Signal:
        strategy_name = self.__class__.__name__

        if df.empty:
            return Signal(
                direction=Direction.LONG,
                strength=None,
                date=None,
                strategy=strategy_name,
                metadata={"error": "Empty DataFrame."}
            )

        try:
            last = df.iloc[-1]

            return self._score(
                current_date=df.index[-1],
                strategy_name=strategy_name,
                rsi_prev=df.iloc[-2]["rsi"],
                rsi=last["rsi"]
            )

        except Exception as exc:
            return Signal(
                direction=Direction.INVALID,
                strength=None,
                date=None,
                strategy=strategy_name,
                metadata={"error": str(exc)}
            )

    def _score(self, *, current_date: date, strategy_name: str, rsi_prev: float, rsi: float) -> Signal:

        #
        # BUY
        #
        rsi_up_threshold = int(self.params.get("rsi_up_threshold", 40))
        if (rsi_prev < rsi_up_threshold) & (rsi >= rsi_up_threshold):
            return Signal(
                direction=Direction.LONG,
                strength=((rsi / rsi_prev) - 1.0),
                date=current_date,
                strategy=strategy_name
            )

        #
        # SELL
        #
        rsi_down_threshold = int(self.params.get("rsi_down_threshold", 60))
        if (rsi_prev > rsi_down_threshold) & (rsi <= rsi_down_threshold):
            return Signal(
                direction=Direction.SHORT,
                strength=(1.0 - (rsi / rsi_prev)),
                date=current_date,
                strategy=strategy_name
            )

        return Signal(
            direction=Direction.FLAT,
            strength=None,
            date=current_date,
            strategy=strategy_name
        )
