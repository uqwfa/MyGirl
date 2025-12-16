import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class Proxy:
    id: int
    isin: str
    name: str
    exchange_id: int
    last_date: pd.Timestamp
