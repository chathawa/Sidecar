from __future__ import annotations
from functools import lru_cache
from typing import Tuple

import numpy
import numpy as np
import pandas as pd
from scipy.stats import norm

from sidecar.prices import *

DEFAULT_NUM_STEPS = 100


CDFColumns = Tuple[
    np.ndarray,  # x steps
    np.ndarray,  # cdf
    np.ndarray   # ecdf
]


def x_steps(changes: FloatColumn, num_steps=DEFAULT_NUM_STEPS):
    start, stop = min(changes), max(changes)
    return np.arange(start, stop, (stop - start) / num_steps)


def ecdf(x: np.ndarray, changes: np.ndarray):
    return np.array([
        np.where(changes <= value)[0].shape[0] / x.shape[0] for value in x
    ])


def cdf(x: np.ndarray, changes: np.ndarray, ecdf_y: np.ndarray):
    return max(ecdf_y) * norm.cdf(x, loc=changes.mean(), scale=changes.std())


def from_price_changes(changes: FloatColumn, num_steps=DEFAULT_NUM_STEPS) -> CDFColumns:
    x = x_steps(changes, num_steps=num_steps)
    ecdf_y = ecdf(x, changes)

    return x, cdf(x, changes, ecdf_y), ecdf_y


def from_prices_data(prices: PricesDataset, num_steps=DEFAULT_NUM_STEPS, log_changes=True) -> CDFColumns:
    return from_price_changes(prices.log_changes if log_changes else prices.changes, num_steps=num_steps)


class CDFDataset(Dataset):
    _x: np.ndarray
    _cdf: np.ndarray
    _ecdf: np.ndarray

    def __init__(
            self,
            x: FloatColumn,
            cdf_y: FloatColumn,
            ecdf_y: FloatColumn
    ):
        self._x, self._cdf, self._ecdf = (
            self.column_to_array(column) for column in (x, cdf_y, ecdf_y)
        )

    @property
    @lru_cache(1)
    def x(self):
        return self._x.copy()

    @property
    @lru_cache(1)
    def cdf(self):
        return self._cdf.copy()

    @property
    @lru_cache(1)
    def ecdf(self):
        return self._ecdf.copy()

    @lru_cache(1)
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(data={
            "changes": self._x,
            "cdf": self._cdf,
            "ecdf": self._ecdf
        })

    @classmethod
    def from_price_changes(cls, changes: FloatColumn, num_steps=DEFAULT_NUM_STEPS) -> CDFDataset:
        return cls(*from_price_changes(changes, num_steps=num_steps))

    @classmethod
    def from_prices_data(cls, prices: PricesDataset, num_steps=DEFAULT_NUM_STEPS) -> CDFDataset:
        return cls(*from_prices_data(prices, num_steps=num_steps))
