import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PositionContext:
    entry_price: float
    current_maximum_price: float
    days_held: int


class BacktesterStrategy(ABC):

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def check_buy(self, row: pd.Series) -> bool:
        pass

    @abstractmethod
    def check_sell(self, row: pd.Series, context: PositionContext) -> bool:
        pass


class ThresholdStrategy(ABC):
    @property
    @abstractmethod
    def lookback_period(self) -> int:
        pass

    @abstractmethod
    def get_buy_mask(self, history: np.ndarray, hypo_prices: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def get_sell_mask(self, history: np.ndarray, hypo_prices: np.ndarray, **kwargs) -> np.ndarray:
        pass
