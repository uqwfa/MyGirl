from datetime import date

import pandas as pd

from strategy.models import Signal, Direction
from strategy.strategies.base import BaseStrategy


class BuyAndHold(BaseStrategy):
    """"""

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signal(self, df: pd.DataFrame, *, buy_date: date | None = None) -> Signal:
        return Signal(
            direction=Direction.LONG,
            strength=1.0,
            date=df.index[-1],
            strategy=self.__class__.__name__,
            metadata={}
        )
