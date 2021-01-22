import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger()


class ImputerError(Exception):
    """Exception raised for errors in Imputers"""
    pass


class BaseImputer(ABC):

    def __init__(self):
        self._fitted = False

    def _guess_dtypes(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        categorical_columns = [c for c in data.columns if pd.api.types.is_categorical_dtype(data[c])]
        numerical_columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and c not in categorical_columns]
        return categorical_columns, numerical_columns

    def _restore_dtype(self, data: pd.DataFrame, dtypes: pd.Series) -> None:
        for column in data.columns:
            data[column] = data[column].astype(dtypes[column].name)

    @abstractmethod
    def fit(self, data: pd.DataFrame, target_columns: List[str], refit: bool = False):

        self._target_columns = target_columns
        self._categorical_columns, self._numerical_columns = self._guess_dtypes(data)

        # some basic error checking
        if self._fitted and not refit:
            raise ImputerError("Imputer is already fitted. Force refitting with 'refit'.")

        if not type(target_columns) == list:
            raise ImputerError(f"Parameter 'target_column' need to be of type list but is '{type(target_columns)}'")

        if any([column not in data.columns for column in self._target_columns]):
            raise ImputerError(f"All target columns ('{self._target_columns}') must be in: {', '.join(data.columns)}")

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        pass


class SklearnBaseImputer(BaseImputer):

    def __init__(
        self,
        categorical_imputer: Tuple[BaseEstimator, Dict[str, object]],
        numerical_imputer: Tuple[BaseEstimator, Dict[str, object]],
        categorical_precision_threshold: float = 0.85,
        encode_as: str = "one-hot"
    ):

        valid_encodings = ["one-hot", "ordinal"]

        if encode_as not in valid_encodings:
            raise ImputerError(f"parameter 'encode_as' need to be one of: {', '.join(valid_encodings)}")

        self._encoder = OneHotEncoder(handle_unknown='ignore') if encode_as == "one-hot" else OrdinalEncoder(handle_unknown='ignore')

        super().__init__()

        self._predictors: Dict[str, BaseEstimator] = {}
        self._categorical_precision_threshold = categorical_precision_threshold
        self._numerical_imputer = (
            numerical_imputer[0],
            {f'numerical_imputer__{name}': value for name, value in numerical_imputer[1].items()}
        )
        self._categorical_imputer = (
            categorical_imputer[0],
            {f'categorical_imputer__{name}': value for name, value in categorical_imputer[1].items()}
        )

    @staticmethod
    def _categorical_columns_to_string(data_frame: pd.DataFrame) -> pd.DataFrame:
        missing_mask = data_frame.isna()

        for column in data_frame.columns:
            if pd.api.types.is_categorical_dtype(data_frame[column]):
                data_frame[column] = data_frame[column].astype(str)

        # preserve missing values
        data_frame[missing_mask] = np.nan
        return data_frame

    def _get_pipeline_and_parameters(self, column: str) -> Tuple[Pipeline, Dict[str, object]]:

        # define general pipeline for processing columns this will be applied on all variables
        # that are use for prediction, i.e. predictor variables
        categorical_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='constant', fill_value='__NA__')),
                ('encode', self._encoder)
            ]
        )

        numeric_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='mean')),  # NOTE: for paper: make clear we use mean imputation without further optimization.
                ('scale',  StandardScaler())
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
                    ('learner', self._categorical_imputer[0])
                ]
            )

            parameters = self._categorical_imputer[1]

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
                    ('learner', self._numerical_imputer[0])
                ]
            )

            parameters = self._numerical_imputer[1]

        return pipeline, parameters

    def fit(self, data: pd.DataFrame, target_columns: List[str], refit: bool = False) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns, refit=refit)

        # cast categorical columns to strings fixes problems where categories are integer values and treated as regression task
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for column in self._target_columns:
            # NOTE: We use the whole available data.
            # If there are missing values in predictor columns, they getting imputed beforehand to use them for fitting.

            pipeline, parameters = self._get_pipeline_and_parameters(column)
            search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)

            # NOTE: target column is excluded in the pipeline. So, wouldn't be used of fit/predict.
            self._predictors[column] = search.fit(data, data[column]).best_estimator_
            logger.debug(f"Predictor for column '{column}' reached {search.best_score_}")

        self._fitted = True

        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

        # save the original dtypes because ..
        dtypes = data.dtypes

        # ... dtypes of data need to be same as for fitting
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        imputed_mask = data[self._target_columns].isna().any(axis=1)

        for column in self._target_columns:
            missing_mask = data[column].isna()
            amount_missing_in_columns = missing_mask.sum()

            if amount_missing_in_columns > 0:
                data.loc[missing_mask, column] = self._predictors[column].predict(data.loc[missing_mask])

                logger.debug(f'Imputed {amount_missing_in_columns} values in column {column}')

        self._restore_dtype(data, dtypes)

        return data, imputed_mask
