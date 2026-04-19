"""
strategy/indicators/bollingerBands.py
-------------------------------------
"""

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
