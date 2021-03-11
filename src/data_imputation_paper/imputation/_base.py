import logging
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .utils import set_seed

logger = logging.getLogger()
warnings.filterwarnings("ignore")


class ImputerError(Exception):
    """Exception raised for errors in Imputers"""
    pass


class BaseImputer(ABC):

    def __init__(self, seed: Optional[int] = None):
        """
        Abstract Base Class that defines the interface for all Imputer classes.

        Args:
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        self._fitted = False
        self._seed = seed

        set_seed(self._seed)

    @staticmethod
    def _guess_dtypes(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Helper method: finds categorical and numerical columns.

        Args:
            data (pd.DataFrame): Data to guess the columns data types

        Returns:
            Tuple[List[str], List[str]]: Lists of categorical and numerical column names
        """

        categorical_columns = [c for c in data.columns if pd.api.types.is_categorical_dtype(data[c])]
        numerical_columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and c not in categorical_columns]
        return categorical_columns, numerical_columns

    @staticmethod
    def _categorical_columns_to_string(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Treats the categorical columns as strings and preserves missing values.

        Args:
            data_frame (pd.DataFrame): To-be-converted data

        Returns:
            pd.DataFrame: Data, where the categorical columns are strings
        """

        missing_mask = data_frame.isna()

        for column in data_frame.columns:
            if pd.api.types.is_categorical_dtype(data_frame[column]):
                data_frame[column] = data_frame[column].astype(str)

        # preserve missing values
        data_frame[missing_mask] = np.nan
        return data_frame

    def _restore_dtype(self, data: pd.DataFrame, dtypes: pd.Series) -> None:
        """
        Restores the data types of the columns

        Args:
            data (pd.DataFrame): Data, which column data types need to be restored
            dtypes (pd.Series): Data types
        """

        for column in data.columns:
            data[column] = data[column].astype(dtypes[column].name)

    @abstractmethod
    def get_best_hyperparameters(self) -> dict:
        """
        Returns the hyperparameters found as best during fitting.

        Returns:
            dict: Best hyperparameters
        """

        if not self._fitted:
            raise ImputerError("Imputer is not fitted.")

        return {}

    @abstractmethod
    def fit(self, data: pd.DataFrame, target_columns: List[str]):
        """
        Fit the imputer based on given `data` to imputed the `target_columns` later on.

        Args:
            data (pd.DataFrame): Data to train the imputer on.
            target_columns (List[str]): To-be-imputed columns.

        Raises:
            ImputerError: If `target_columns` is not a list.
            ImputerError: If element of `target_columns` is not column of `data`.
        """

        # some basic error checking
        if self._fitted:
            raise ImputerError(f"Imputer is already fitted. Target columns: {', '.join(self._target_columns)}")

        if not type(target_columns) == list:
            raise ImputerError(f"Parameter 'target_column' need to be of type list but is '{type(target_columns)}'")

        if any([column not in data.columns for column in target_columns]):
            raise ImputerError(f"All target columns ('{target_columns}') must be in: {', '.join(data.columns)}")

        self._target_columns = target_columns
        self._categorical_columns, self._numerical_columns = self._guess_dtypes(data)

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Imputes the columns of (the copied) `data` the imputer is fitted for.

        Args:
            data (pd.DataFrame): To-be-imputed data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: First return value (index 0) is the (copied and) imputed `data`. \
                Second return value (index 1) is a mask representing which values are imputed. \
                It is a `DataFrame` because argument `target_columns` for `fit` method uses `list` of column names.
        """

        # some basic error checking
        if not self._fitted:
            raise ImputerError("Imputer is not fitted.")


class SklearnBaseImputer(BaseImputer):

    def __init__(
        self,
        categorical_imputer: Tuple[BaseEstimator, Dict[str, object]],
        numerical_imputer: Tuple[BaseEstimator, Dict[str, object]],
        encode_as: str = "one-hot",
        seed: Optional[int] = None
    ):
        """
        Base class for scikit-learn based imputers. Builds for each to-be-imputed column a pipeline \
            that incorporates all other columns for imputation.

        Args:
            categorical_imputer (Tuple[BaseEstimator, Dict[str, object]]): imputer object and hyperparameter grid for categorical columns
            numerical_imputer (Tuple[BaseEstimator, Dict[str, object]]): imputer object and hyperparameter grid for numerical columns
            encode_as (str, optional): Defines how to encode categorical columns. Defaults to "one-hot".
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.

        Raises:
            ImputerError: [description]
        """

        valid_encodings = ["one-hot", "ordinal"]

        if encode_as not in valid_encodings:
            raise ImputerError(f"parameter 'encode_as' need to be one of: {', '.join(valid_encodings)}")

        self._encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) if encode_as == "one-hot" else OrdinalEncoder(handle_unknown='ignore')

        super().__init__(seed=seed)

        self._predictors: Dict[str, BaseEstimator] = {}
        self._numerical_imputer = (
            numerical_imputer[0],
            {f'numerical_imputer__{name}': value for name, value in numerical_imputer[1].items()}
        )
        self._categorical_imputer = (
            categorical_imputer[0],
            {f'categorical_imputer__{name}': value for name, value in categorical_imputer[1].items()}
        )

    def _get_pipeline_and_hyperparameter_grid(self, column: str) -> Tuple[Pipeline, Dict[str, object]]:
        """
        Helper method: builds the imputer pipeline for a given column.

        Args:
            column (str): Target column

        Returns:
            Tuple[Pipeline, Dict[str, object]]: Imputer pipeline and hyperparameter grid
        """

        # define general pipeline for processing columns this will be applied on all variables
        # that are use for prediction, i.e. predictor variables
        categorical_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='most_frequent')),
                ('encode', self._encoder)
            ]
        )

        numeric_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='mean')),
            ]
        )

        # (to-be-imputed-)column is categorical ..
        if column in self._categorical_columns:

            # .. so we need to remove the target (column) from the list of categorical columns
            categorical_predictor_variables = [x for x in self._categorical_columns if x != column]

            feature_transformation = ColumnTransformer(transformers=[
                    ('categorical_features', categorical_preprocessing, categorical_predictor_variables),
                    ('numerical_features', numeric_preprocessing, self._numerical_columns)
                ]
            )

            pipeline = Pipeline(
                [
                    ('features', feature_transformation),
                    ('scale',  StandardScaler()),
                    ('categorical_imputer', self._categorical_imputer[0])
                ]
            )

            hyperparameter_grid = self._categorical_imputer[1]

        # (to-be-imputed-)column is numerical ...
        elif column in self._numerical_columns:

            # ... so we need to remove the target (column) from the list of numerical columns
            numerical_predictor_variables = [x for x in self._numerical_columns if x != column]

            feature_transformation = ColumnTransformer(transformers=[
                    ('categorical_features', categorical_preprocessing, self._categorical_columns),
                    ('numerical_features', numeric_preprocessing, numerical_predictor_variables)
                ]
            )

            pipeline = Pipeline(
                [
                    ('features', feature_transformation),
                    ('scale',  StandardScaler()),
                    ('numerical_imputer', self._numerical_imputer[0])
                ]
            )

            hyperparameter_grid = self._numerical_imputer[1]

        return pipeline, hyperparameter_grid

    def fit(self, data: pd.DataFrame, target_columns: List[str]) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns)

        # cast categorical columns to strings fixes problems where categories are integer values and treated as regression task
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for column in self._target_columns:
            # NOTE: We use the whole available data.
            # If there are missing values in predictor columns, they getting imputed (marked) beforehand to use them for fitting.

            pipeline, hyperparameter_grid = self._get_pipeline_and_hyperparameter_grid(column)
            search = GridSearchCV(pipeline, hyperparameter_grid, cv=5, n_jobs=-1)

            missing_mask = data[column].isna()
            self._predictors[column] = search.fit(data[~missing_mask], data.loc[~missing_mask, column]).best_estimator_
            logger.debug(f"Predictor for column '{column}' reached {search.best_score_}")

        self._fitted = True

        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        super().transform(data=data)

        imputed_mask = data[self._target_columns].isna()

        # save the original dtypes because ..
        dtypes = data.dtypes

        # ... dtypes of data need to be same as for fitting
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for column in self._target_columns:
            missing_mask = data[column].isna()
            amount_missing_in_columns = missing_mask.sum()

            if amount_missing_in_columns > 0:
                data.loc[missing_mask, column] = self._predictors[column].predict(data.loc[missing_mask])

                logger.debug(f'Imputed {amount_missing_in_columns} values in column {column}')

        self._restore_dtype(data, dtypes)

        return data, imputed_mask

    def get_best_hyperparameters(self) -> dict:

        super().get_best_hyperparameters()

        best_hyperparameters = dict()

        for column in self._target_columns:
            hyperparameters = [
                (hyperparameter, value)
                for hyperparameter, value in self._predictors[column].get_params().items()
                if "_imputer__" in hyperparameter
            ]

            # remove leading strings that come from sklearn pipeline
            best_hyperparameters[column] = {hyperparameter[hyperparameter.find("__") + 2:]: value for hyperparameter, value in hyperparameters}

        return best_hyperparameters
