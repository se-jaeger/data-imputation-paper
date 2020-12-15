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
        self._imputer = None
        self._data_encoder = None

    def _guess_dtypes(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        categorical_columns = [c for c in data.columns if pd.api.types.is_categorical_dtype(data[c])]
        numerical_columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and c not in categorical_columns]
        return categorical_columns, numerical_columns

    def _is_regression_imputation(self, data: pd.DataFrame, target_column: str) -> bool:
        return pd.api.types.is_numeric_dtype(data[target_column])

    def _is_classification_imputation(self, data: pd.DataFrame, target_column: str) -> bool:
        return pd.api.types.is_categorical_dtype(data[target_column])

    def _is_fitted(self, estimator: BaseEstimator) -> bool:
        return_value = False

        if estimator is not None:
            return_value = bool(
                [
                    # this one is the official sklearn to check whether a estimator is fitted
                    v for v in vars(estimator)
                    if (v.endswith("_") or v.startswith("_")) and not v.startswith("__")
                ]
            )

        return return_value

    @abstractmethod
    def _encode_data_for_imputation(self, data: pd.DataFrame, refit: bool = False) -> Tuple[pd.DataFrame, list]:
        pass

    @abstractmethod
    def _decode_data_after_imputation(self, imputed: pd.array, index: pd.Index, dtypes: pd.Series) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame, target_column: str, refit: bool = False, **kwargs) -> None:

        self._target_column = target_column

        # some basic error checking
        if self._fitted and not refit:
            raise ImputerError("Imputer is already fitted. Force refitting with 'refit'.")

        if self._target_column not in data.columns:
            raise ImputerError(f"target column '{target_column}' not found, must be one of: {', '.join(data.columns)}")

    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, list]:
        pass


class SKLearnModeImputer(BaseImputer):

    def __init__(self, imputer_args: dict = {}):

        from sklearn.impute import SimpleImputer

        super().__init__()

        self._imputer = SimpleImputer(**imputer_args)

    def fit(self, data: pd.DataFrame, target_column: str, refit: bool = False, **kwargs) -> None:

        super().fit(data=data, target_column=target_column, refit=refit)

        # set proper strategy for column type
        if self._is_regression_imputation(data, target_column):
            self._imputer.set_params(strategy="mean")

        elif self._is_classification_imputation(data, target_column):
            self._imputer.set_params(strategy="most_frequent")

        self._imputer.fit(data[target_column].to_numpy().reshape(-1, 1), **kwargs)  # sklearns' SimpleImputer expects 2D array

        self._fitted = True

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

        completed = data.copy()
        imputed_mask = completed[self._target_column].isnull()  # TODO: does this work in all cases

        # sklearns' SimpleImputer expects 2D array
        imputed = self._imputer.transform(completed[self._target_column].to_numpy().reshape(-1, 1))
        completed[self._target_column] = imputed

        return completed, imputed_mask

