import pandas as pd
import numpy as np
from typing import Callable


class ThresholdCalculator:
    @staticmethod
    def calculate_thresholds(df: pd.DataFrame, target: pd.Timestamp, strat_func: Callable, window_size: int,
                             num_steps: int = 100, scale: float = 0.1, column: str = "close",
                             refinement_steps: int = 10, **kwargs) -> tuple[list[list[float]], pd.Timestamp]:
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

        is_current_true = results_mask[0]
        curr_start = prices[0] if is_current_true else None

        for i in range(len(results_mask) - 1):
            state_now = results_mask[i]
            state_next = results_mask[i + 1]

            if state_now == False and state_next == True:
                # Transition False -> True
                refined_start = ThresholdCalculator._binary_search(
                    history, prices[i], prices[i + 1], strat_func, True, refinement_steps, **kwargs
                )

                curr_start = refined_start

            elif state_now == True and state_next == False:
                # Transition True -> False
                refined_end = ThresholdCalculator._binary_search(
                    history, prices[i], prices[i + 1], strat_func, False, refinement_steps, **kwargs
                )

                if curr_start is not None:
                    intervals.append([curr_start, refined_end])
                    curr_start = None

        if curr_start is not None:
            intervals.append([curr_start, prices[-1]])

        return intervals, reference_date

    @staticmethod
    def _binary_search(history: np.ndarray, low_p: float, high_p: float, strat_func: Callable, target_state: bool,
                       iterations: int = 10, **kwargs) -> float:

        for _ in range(iterations):
            mid_p = (low_p + high_p) / 2
            res = strat_func(history, np.array([mid_p]), **kwargs)[0]

            if res == target_state:
                high_p = mid_p

            else:
                low_p = mid_p

        return (low_p + high_p) / 2
