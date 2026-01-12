import pandas as pd
import numpy as np
from typing import Callable


class ThresholdCalculator:
    @staticmethod
    def calculate_thresholds(df: pd.DataFrame, target: pd.Timestamp, strat_func: Callable, window_size: int,
                             num_steps: int = 1000, scale: float = 0.1, column: str = "close", **kwargs) -> tuple[list[list[float]], pd.Timestamp]:
        if df.empty:
            print("Empty DataFrame.")
            return [], target

        if target in df.index:
            reference_date = target
            loc_idx = df.index.get_loc(target)
            original_price = df.at[target, column]

        else:
            last_date = df.index[-1]

            if target < last_date:
                print(f"Target date {target} missing from index (Gap).")
                return [], target

            reference_date = last_date
            loc_idx = len(df)
            original_price = df[column].iloc[-1]

        start_loc = loc_idx - window_size
        if start_loc < 0:
            print("Not enough historical data to perform calculation.")
            return [], reference_date

        history = df[column].iloc[start_loc : loc_idx].to_numpy()

        prices = np.linspace(
            original_price * (1 - scale),
            original_price * (1 + scale),
            num_steps
        )

        results_mask = strat_func(history, prices, **kwargs)

        intervals = []
        curr_start = None

        for price, is_true in zip(prices, results_mask):
            if is_true:
                if curr_start is None:
                    curr_start = price

            else:
                if curr_start is not None:
                    intervals.append([curr_start, price])
                    curr_start = None

        if curr_start is not None:
            intervals.append([curr_start, prices[-1]])

        return intervals, reference_date
