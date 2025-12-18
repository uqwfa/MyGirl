import pandas as pd

from modules.technical_analysis.threshold import ThresholdCalculator
from modules.simulation.objects.security import Security
from modules.technical_analysis.book import BookStrategy


class TechnicalAnalysisCore:

    def get_stats(self, securities: list[Security]) -> dict[Security, dict]:
        return {
            sec: self.process_sec(sec)
            for sec in securities
        }

    @staticmethod
    def process_sec(security: Security) -> dict:
        data = security.data
        target_request = pd.Timestamp.today()

        buy_intervals, used_date = ThresholdCalculator.calculate_thresholds(
            df=data,
            target=target_request,
            strat_func=BookStrategy.get_buy_mask
        )

        sell_intervals, used_date = ThresholdCalculator.calculate_thresholds(
            df=data,
            target=target_request,
            strat_func=BookStrategy.get_sell_mask
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
