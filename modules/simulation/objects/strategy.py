import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PositionContext:
    entry_price: float
    current_maximum_price: float
    days_held: int


class Strategy(ABC):

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def check_buy(self, row: pd.Series) -> bool:
        pass

    @abstractmethod
    def check_sell(self, row: pd.Series, context: PositionContext) -> bool:
        pass
