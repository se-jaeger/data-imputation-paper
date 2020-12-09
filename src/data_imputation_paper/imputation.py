from abc import ABC, abstractmethod

from typing import Tuple

import numpy as np
import pandas as pd


class ImputerError(Exception):
    """Exception raised for errors in Imputers"""
    pass


class BaseImputer(ABC):

    def __init__(self):
        self._fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame, target_column: str, columns: list = [], refit: bool = False, **kwargs) -> None:

        # some basic error checking
        if self._fitted and not refit:
            raise ImputerError("Imputer is already fitted. Force refitting with 'refit'.")

        if target_column not in data.columns:
            raise ImputerError(f"target column '{target_column}' must be one of: {', '.join(data.columns)}")

    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        pass


class SKLearnModeImputer(BaseImputer):

    def fit(self, data: pd.DataFrame, target_column: str, columns: list = [], refit: bool = False, **kwargs) -> None:

        super().fit(data=data, target_column=target_column, columns=columns, refit=refit)

        from sklearn.impute import SimpleImputer

        # mean imputation error handling
        if pd.api.types.is_numeric_dtype(data[target_column]):
            self._imputer = SimpleImputer(strategy="mean", **kwargs)

        # mode imputation error handling
        elif pd.api.types.is_categorical_dtype(data[target_column]):
            self._imputer = SimpleImputer(strategy="most_frequent", **kwargs)

        self._target_column = target_column
        self._columns = columns
        self._imputer.fit(data[target_column].to_numpy().reshape(-1, 1))  # sklearn expects 2D array

        self._fitted = True

    def transform(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:

        completed = data.copy()
        imputed_mask = data[self._target_column].isnull()  # TODO: does this work in all cases

        # sklearn expects 2D array
        imputed = self._imputer.transform(completed[self._target_column].to_numpy().reshape(-1, 1))
        completed[self._target_column] = imputed

        return completed, imputed_mask
