"""
strategy/indicators/bollingerBands.py
-------------------------------------
"""

import numpy as np
import pandas as pd


def add_bb(df: pd.DataFrame, window: int, factor: float = 2.0) -> pd.DataFrame:
    """Add Bollinger bands columns 'bb_upper' and 'bb_lower' to a dataframe and return it."""

    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")
    if window > len(df):
        raise ValueError(f"window ({window}) exceeds the number of rows in df ({len(df)}).")

    rolling = df["close"].rolling(window=window)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()

    df["bb_upper"] = rolling_mean + (factor * rolling_std)
    df["bb_lower"] = rolling_mean - (factor * rolling_std)

    return df


def bb_at_price(history: np.ndarray, test_price: float | np.ndarray, window: int, factor: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute Bollinger Band values for one or many hypothetical closing prices."""

    if len(history) != window - 1:
        raise ValueError(f"history_close must have exactly window-1={window - 1} elements, got {len(history)}.")

    test_price = np.asarray(test_price, dtype=float)
    scalar_input = test_price.ndim == 0
    test_price = np.atleast_1d(test_price)

    # shape: (n_prices, window)
    window_matrix = np.column_stack(
        [np.tile(history, (len(test_price), 1)), test_price]
    )

    means = window_matrix.mean(axis=1)
    stds = window_matrix.std(axis=1, ddof=1)

    bb_upper = means + (factor * stds)
    bb_lower = means - (factor * stds)

    if scalar_input:
        return float(bb_upper[0]), float(bb_lower[0])

    return bb_upper, bb_lower
