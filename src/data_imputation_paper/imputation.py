from abc import ABC, abstractmethod
from typing import Tuple, List

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # only imported but will not be used explicitely
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

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

        super().__init__()

        self._imputer_args = imputer_args
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

    def transform(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:

        completed = data.copy()
        imputed_mask = completed[self._target_column].isnull()  # TODO: does this work in all cases

        # sklearns' SimpleImputer expects 2D array
        imputed = self._imputer.transform(completed[self._target_column].to_numpy().reshape(-1, 1), **kwargs)
        completed[self._target_column] = imputed

        return completed, imputed_mask

    def _encode_data_for_imputation(self, data: pd.DataFrame, refit: bool = False) -> Tuple[pd.DataFrame, list]:
        pass

    def _decode_data_after_imputation(self, imputed: pd.array, index: pd.Index, dtypes: pd.Series) -> pd.DataFrame:
        pass


class SKLearnIterativeImputer(BaseImputer):

    _IMPLEMENTED_STRATEGIES = ["MissForest".lower(), "MICE".lower()]

    def __init__(self, strategy: str, data_encoding_type: str = "one-hot", imputer_args: dict = {}, estimator_args: dict = {}):

        super().__init__()

        self._strategy = strategy.lower()
        self._imputer_args = imputer_args
        self._estimator_args = estimator_args

        if data_encoding_type in ["one-hot", "ordinal"]:
            self._data_encoding_type = data_encoding_type
        else:
            raise ImputerError(f"don't know how to decode data for type '{self._data_encoding_type}'")

        if "forest" in self._strategy:
            estimator = RandomForestRegressor(**estimator_args)

        elif "mice" in self._strategy:
            estimator = BayesianRidge(**estimator_args)

        else:
            raise ImputerError(f"given strategy '{strategy}' is not implemented. Need to be one of: {', '.join(self._IMPLEMENTED_STRATEGIES)}")

        self._imputer = IterativeImputer(estimator=estimator, **imputer_args)

    def fit(self, data: pd.DataFrame, target_column: str, refit: bool = False, **kwargs) -> None:

        super().fit(data=data, target_column=target_column, refit=refit)

        encoded_data, _ = self._encode_data_for_imputation(data, refit=True)

        self._imputer.fit(encoded_data, **kwargs)
        self._fitted = True

    def transform(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, list]:

        encoded_data, imputed_mask = self._encode_data_for_imputation(data)

        # transform returns np.array => need to create a dataframe again
        imputed_array = self._imputer.transform(encoded_data, **kwargs)
        completed_df = self._decode_data_after_imputation(imputed_array, data.index, data.dtypes)

        # fix variable order and return imputed data and mask
        return completed_df[data.columns], imputed_mask

    def _encode_data_for_imputation(self, data: pd.DataFrame, refit: bool = False) -> Tuple[pd.DataFrame, list]:

        # TODO: in sklearn 0.24 OrdinalEncoder will suport `handle_unknown`
        # This hopefully will help to handle the case when the target column is categorical.
        encoder = OneHotEncoder(sparse=False) if self._data_encoding_type == "one-hot" else OrdinalEncoder()

        categorical_columns, numerical_columns = self._guess_dtypes(data)
        mask = list(data[self._target_column].isnull())  # To preserve missing values in target_column

        if refit or not self._is_fitted(self._data_encoder):
            categorical_preprocessing = Pipeline(
                [
                    ('mark_missing', SimpleImputer(strategy='constant', fill_value='__NA__')),
                    ('categorical_encoder', encoder)
                ]
            )

            numerical_preprocessing = Pipeline(
                [
                    ('mark_missing', SimpleImputer(strategy='constant', fill_value=0)),  # is 0 a sensible value?
                ]
            )

            feature_transformation = ColumnTransformer(transformers=[
                    ('categorical_features', categorical_preprocessing, categorical_columns),
                    ('scaled_numeric', numerical_preprocessing, numerical_columns)
                ]
            )

            self._data_encoder = feature_transformation.fit(data)

        encoded_data = self._data_encoder.transform(data)

        return encoded_data, mask

    def _decode_data_after_imputation(self, imputed: pd.array, index: pd.Index, dtypes: pd.Series) -> pd.DataFrame:

        # numerical columns are append on the right.
        # transformer 2 (index 1) is 'numerical_preprocessing', the third value in the tuple represents the column names.
        numerical_columns = self._data_encoder.transformers[1][2]
        categorical_columns = self._data_encoder.transformers[0][2]

        columns = numerical_columns + categorical_columns

        categorical_values = self._data_encoder.named_transformers_["categorical_features"].named_steps["categorical_encoder"] \
            .inverse_transform(imputed[:, :-len(numerical_columns)])  # get one-hot-encoded features and transform them back

        numerical_values = imputed[:, -len(numerical_columns):]

        imputed_df = pd.DataFrame(np.concatenate([numerical_values, categorical_values], axis=1), columns=columns, index=index)

        for column in columns:
            imputed_df[column] = imputed_df[column].astype(dtypes[column].name)

        return imputed_df
