import pandas as pd

from modules.technical_analysis.threshold import ThresholdCalculator
from modules.simulation.objects.strategy import ThresholdStrategy
from modules.simulation.objects.security import Security
from modules.technical_analysis.book import BookStrategy
from modules.simulation.objects.order import Order


class TechnicalAnalysisCore:
    def __init__(self, strategy: ThresholdStrategy):
        self.strategy = strategy

    def get_stats(self, securities: list[Security | Order]) -> dict:
        return {
            sec: self.process_sec(sec) if isinstance(sec, Security) else self.process_order(sec)
            for sec in securities
        }

    def process_order(self, order: Order) -> dict:
        return self.process_sec(order.sec, maximum=order.get_maximum(pd.Timestamp.today()))

    def process_sec(self, security: Security, **kwargs) -> dict:
        data = security.data
        target_request = pd.Timestamp.today()

        lookback = self.strategy.lookback_period

        buy_intervals, used_date = ThresholdCalculator.calculate_thresholds(
            df=data,
            target=target_request,
            strat_func=self.strategy.get_buy_mask,
            window_size=lookback,
            **kwargs
        )

        sell_intervals, used_date = ThresholdCalculator.calculate_thresholds(
            df=data,
            target=target_request,
            strat_func=self.strategy.get_sell_mask,
            window_size=lookback,
            **kwargs
        )

        if used_date in data.index:
            curr_price = data.at[used_date, "close"]

        else:
            curr_price = 0.0

        is_forecast = target_request.normalize() > used_date.normalize()

        return {
            "status": TechnicalAnalysisCore.get_status(buy_intervals, sell_intervals, curr_price),
            "requested_date": target_request,
            "used_date": used_date,
            "is_forecast": is_forecast,
            "price": curr_price,
            "buy_interval": buy_intervals,
            "sell_interval": sell_intervals
        }

    @staticmethod
    def get_status(buy_intervals, sell_intervals, price) -> str:
        in_buy = any(start <= price <= end for start, end in buy_intervals)
        in_sell = any(start <= price <= end for start, end in sell_intervals)

        if in_buy:
            return "buy"

        elif in_sell:
            return "sell"

        else:
            return "hold"
