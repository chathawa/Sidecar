from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Union, Type, Dict, Any
from datetime import date

import pandas as pd
import numpy as np


class Dataset(ABC):
    _DF_FILE_EXTENSION = 'pkl'
    _pickled_df_name: str

    @staticmethod
    def column_to_array(
            column: Union[Iterable, np.ndarray],
            dtype: Type = np.float32
    ) -> np.ndarray:
        return column.copy() if isinstance(column, np.ndarray) else np.array(column, dtype=dtype)

    @abstractmethod
    def to_df(self, *args, **kwargs) -> pd.DataFrame:
        ...

    @classmethod
    @abstractmethod
    def from_df(cls, df: pd.DataFrame) -> Dataset:
        ...

    @property
    @abstractmethod
    def df_name_kwargs(self) -> Dict[str, Any]:
        ...

    def dump(self, path: Path, *args, name=None, **kwargs):
        if name is None:
            name = f"{self._pickled_df_name.format(**self.df_name_kwargs)}.{self._DF_FILE_EXTENSION}"

        self.to_df(*args, **kwargs).to_pickle(str(path.joinpath(name)))

    @classmethod
    def load(cls, path: Path) -> Dataset:
        return cls.from_df(pd.read_pickle(path))


DateColumn = Union[Iterable[date], np.ndarray]
FloatColumn = Union[Iterable[float], np.ndarray]


