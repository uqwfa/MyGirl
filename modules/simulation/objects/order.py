import pandas as pd

from modules.simulation.objects.security import Security


class Order:
    def __init__(self, security: Security, capital: float, buy_date: pd.Timestamp, buy_price: float = None):
        self._sec = security
        self.start_capital = capital
        self.buy_date = buy_date

        if buy_price is None:
            if buy_date in security.data.index:
                self.buy_price = security.data.at[buy_date, "close"]

            else:
                self.buy_price = -1

        else:
            self.buy_price = buy_price

    @property
    def sec(self) -> Security:
        return self._sec

    def get_maximum(self, end_date: pd.Timestamp) -> float:
        data = self._sec.data

        window = data.loc[self.buy_date : end_date]

        if window.empty:
            return 0.0

        return float(window["close"].max())
