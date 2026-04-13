"""
ingestion/normalizer.py
-----------------------
Converts raw collector DataFrames into OHLCVRow objects ready for storage.
"""

import math
import pandas as pd
from typing import Callable

from storage.models import OHLCVRow


def _coerce_float(value) -> float | None:
    """Return None for NaN / None, otherwise the float value."""

    if value is None:
        return None

    try:
        f = float(value)
        return None if math.isnan(f) else f

    except (TypeError, ValueError):
        return None


def _coerce_int(value) -> int | None:
    """Return None for NaN / None, otherwise the int value."""

    f = _coerce_float(value)
    return None if f is None else int(f)


def _normalize_ariva(df: pd.DataFrame, isin: str) -> list[OHLCVRow]:
    """Normalize data returned by ArivaScraper."""

    rows: list[OHLCVRow] = []

    for row_date, row in df.iterrows():
        close = _coerce_float(row.get("close"))

        if close is None:
            continue

        rows.append(OHLCVRow(
            isin=isin,
            date=row_date,
            open=_coerce_float(row.get("open")),
            high=_coerce_float(row.get("high")),
            low=_coerce_float(row.get("low")),
            close=close,
            volume=_coerce_int(row.get("volume"))
        ))

    return rows


_NORMALIZERS: dict[str, Callable] = {
    "ArivaScraper": _normalize_ariva,
}


def normalize(df: pd.DataFrame, isin: str, collector: str) -> list[OHLCVRow]:
    """Convert a raw collector DataFrame into a list of OHLCVRow objects."""

    if df.empty:
        return []

    normalizer_fn = _NORMALIZERS.get(collector)
    if normalizer_fn is None:
        registered = ", ".join(_NORMALIZERS.keys())

        raise ValueError(
            f"No normalizer registered for collector '{collector}'. "
            f"Registered collectors: {registered}"
        )

    return normalizer_fn(df, isin)
