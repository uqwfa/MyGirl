from datetime import date

import optuna
import pandas as pd

from strategy.models import Signal, Direction
from strategy.strategies.base import BaseStrategy


def simple_macro_space(trial: optuna.Trial):
    return {
        "ma_macro_fast": trial.suggest_int("ma_macro_fast", 25, 75),
        "ma_macro_slow": trial.suggest_int("ma_macro_slow", 100, 300)
    }


class SimpleMacroStrategy(BaseStrategy):
    """

    Params (all optional, with defaults for optimization):
        ma_macro_fast  (int, default 50)   - Fast SMA for macro trend filter
        ma_macro_slow  (int, default 200)  - Slow SMA for macro trend filter
    """

    @staticmethod
    def param_space(trial: optuna.Trial):
        return {
            "ma_macro_fast": trial.suggest_int("ma_macro_fast", 25, 75),
            "ma_macro_slow": trial.suggest_int("ma_macro_slow", 100, 300)
        }

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_index()

        ma_macro_fast = int(self.params.get("ma_macro_fast", 50))
        ma_macro_slow = int(self.params.get("ma_macro_slow", 200))
        df["ma_macro_fast"] = df["close"].rolling(ma_macro_fast).mean()
        df["ma_macro_slow"] = df["close"].rolling(ma_macro_slow).mean()

        return df

    def generate_signal(self, df: pd.DataFrame, *, buy_date: date | None = None) -> Signal:
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
                current_date=df.index[-1],
                ma_macro_fast=last["ma_macro_fast"],
                ma_macro_slow=last["ma_macro_slow"],
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

    def _score(self, *, current_date: date, ma_macro_fast: float, ma_macro_slow: float,
               strategy_name: str) -> Signal:

        #
        # BUY
        #
        if ma_macro_fast > ma_macro_slow:
            return Signal(
                direction=Direction.LONG,
                strength=((ma_macro_fast / ma_macro_slow) - 1.0),
                date=current_date,
                strategy=strategy_name
            )

        #
        # SELL
        #
        if ma_macro_fast < ma_macro_slow:
            return Signal(
                direction=Direction.SHORT,
                strength=(1.0 - (ma_macro_fast / ma_macro_slow)),
                date=current_date,
                strategy=strategy_name
            )

        return Signal(
            direction=Direction.FLAT,
            strength=None,
            date=current_date,
            strategy=strategy_name
        )
