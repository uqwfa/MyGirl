import pandas as pd
from typing import Callable, Optional


class Security:
    def __init__(self, id: int, isin: str, name: str, loader: Optional[Callable[[int], pd.DataFrame]] = None):
        self.id = id
        self.isin = isin
        self.name = name

        self._data: Optional[pd.DataFrame] = None
        self._loader = loader

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            if self._loader is None:
                raise ValueError(f"Cannot fetch data for {self.name}: No loader function provided.")

            self._data = self._loader(self.id)

        return self._data

    def refresh_data(self):
        self._data = None
        return self.data
