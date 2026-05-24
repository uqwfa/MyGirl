from datetime import date

import optuna
import pandas as pd

from strategy.models import Signal, Direction
from strategy.strategies.base import BaseStrategy


def simple_agile_space(trial: optuna.Trial):
    return {
        "ema_agile_fast": trial.suggest_int("ema_agile_fast", 4, 14),
        "ema_agile_slow": trial.suggest_int("ema_agile_slow", 11, 31)
    }



class SimpleAgileStrategy(BaseStrategy):
    """
    Params (all optional, with defaults for optimization):
        ema_agile_fast (int, default 9)    - Fast EMA for agile entry/exit trigger
        ema_agile_slow (int, default 21)   - Slow EMA for agile entry/exit trigger
    """

    @staticmethod
    def param_space(trial: optuna.Trial):
        return {
            "ema_agile_fast": trial.suggest_int("ema_agile_fast", 4, 14),
            "ema_agile_slow": trial.suggest_int("ema_agile_slow", 11, 31)
        }

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_index()

        ema_agile_fast = int(self.params.get("ema_agile_fast", 9))
        ema_agile_slow = int(self.params.get("ema_agil_slow", 21))
        df["ema_agile_fast"] = df["close"].ewm(span=ema_agile_fast, adjust=False).mean()
        df["ema_agile_slow"] = df["close"].ewm(span=ema_agile_slow, adjust=False).mean()

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
                ema_agile_fast=last["ema_agile_fast"],
                ema_agile_slow=last["ema_agile_slow"],
                strategy_name=strategy_name
            )

        except Exception as exc:
            return Signal(
                direction=Direction.INVALID,
                strength=None,
                date=None,
                strategy=strategy_name,
                metadata={"error": str(exc)}
            )

    def _score(self, *, current_date: date, ema_agile_fast: float, ema_agile_slow: float,
               strategy_name: str) -> Signal:

        #
        # BUY
        #
        if ema_agile_fast > ema_agile_slow:
            return Signal(
                direction=Direction.LONG,
                strength=((ema_agile_fast / ema_agile_slow) - 1.0),
                date=current_date,
                strategy=strategy_name
            )

        #
        # SELL
        #
        if ema_agile_fast < ema_agile_slow:
            Signal(
                direction=Direction.SHORT,
                strength=(1.0 - (ema_agile_fast / ema_agile_slow)),
                date=current_date,
                strategy=strategy_name
            )

        return Signal(
            direction=Direction.FLAT,
            strength=None,
            date=current_date,
            strategy=strategy_name
        )
