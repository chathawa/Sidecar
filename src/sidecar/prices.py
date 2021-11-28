from __future__ import annotations
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from sidecar.datasets import *
from sidecar.formatting import *


class PricesDataset(Dataset):
    DATE_LABEL = "date"
    CLOSE_LABEL = "close"
    CHANGE_LABEL = "change"

    _pickled_df_name = "{ticker}_{start_date}_{end_date}"

    @property
    def df_name_kwargs(self) -> Dict[str, Any]:
        return {
            "ticker": self._ticker,
            **{
                kwarg: value.strftime(FILENAME_DATE_FORMAT)

                for kwarg, value in (
                    ("start_date", self._start_date),
                    ("end_date", self._end_date)
                )
            }}

    def __init__(
            self,
            ticker: str,
            dates: DateColumn,
            prices: FloatColumn
    ):
        self._ticker = ticker
        self._dates = self.column_to_array(dates, dtype=date)
        self._start_date, self._end_date = min(dates), max(dates)
        self._prices = self.column_to_array(prices)

    @property
    def ticker(self):
        return self._ticker

    @property
    def dates(self):
        return self._dates.copy()

    @property
    def prices(self):
        return self._prices.copy()

    @property
    @lru_cache(1)
    def changes(self):
        return np.array([
            x2 / x1 for x1, x2 in zip(self._prices[:-1], self._prices[1:])
        ], dtype=np.float32)

    @property
    @lru_cache(1)
    def log_changes(self):
        return np.log(self.changes)

    @lru_cache(4)
    def to_df(self, log_changes=True, changes=False) -> pd.DataFrame:
        columns = [
            ("dates", self._dates),
            ("prices", self._prices)
        ]

        if changes:
            columns.append((
                "changes", self.changes
            ))

        if log_changes:
            columns.append((
                "log_changes" if changes else "changes", self.log_changes
            ))

        return pd.DataFrame(data=dict(columns))

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> PricesDataset:
