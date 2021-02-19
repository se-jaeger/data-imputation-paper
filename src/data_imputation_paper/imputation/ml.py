from typing import Dict, Optional, List, Tuple
import logging

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from autokeras import StructuredDataClassifier, StructuredDataRegressor

import pandas as pd

from ._base import SklearnBaseImputer, BaseImputer

logger = logging.getLogger()

class ForestImputer(SklearnBaseImputer):

    def __init__(
        self,
        hyperparameter_grid_categorical_imputer: Dict[str, object] = {},
        hyperparameter_grid_numerical_imputer: Dict[str, object] = {},
        seed: Optional[int] = None
    ):
        """
        Imputer uses `RandomForestClassifier` for categorical columns and `RandomForestRegressor` for numerical columns.

        Args:
            hyperparameter_grid_categorical_imputer (Dict[str, object], optional): \
                Hyperparameter grid for HPO for `RandomForestClassifier`. Defaults to {}.
            hyperparameter_grid_numerical_imputer (Dict[str, object], optional): \
                Hyperparameter grid for HPO for `RandomForestRegressor`. Defaults to {}.
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        super().__init__(
            (RandomForestClassifier(n_jobs=-1), hyperparameter_grid_categorical_imputer),
            (RandomForestRegressor(n_jobs=-1), hyperparameter_grid_numerical_imputer),
            seed=seed
        )


class KNNImputer(SklearnBaseImputer):

    def __init__(
        self,
        hyperparameter_grid_categorical_imputer: Dict[str, object] = {},
        hyperparameter_grid_numerical_imputer: Dict[str, object] = {},
        seed: Optional[int] = None
    ):
        """
        Imputer uses `KNeighborsClassifier` for categorical columns and `KNeighborsRegressor` for numerical columns.

        Args:
            hyperparameter_grid_categorical_imputer (Dict[str, object], optional): \
                Hyperparameter grid for HPO for `KNeighborsClassifier`. Defaults to {}.
            hyperparameter_grid_numerical_imputer (Dict[str, object], optional): \
                Hyperparameter grid for HPO for `KNeighborsRegressor`. Defaults to {}.
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        super().__init__(
            (KNeighborsClassifier(n_jobs=-1), hyperparameter_grid_categorical_imputer),
            (KNeighborsRegressor(n_jobs=-1), hyperparameter_grid_numerical_imputer),
            seed=seed
        )


class AutoKerasImputer(BaseImputer):

    def __init__(
        self,
        max_trials: Optional[int] = 10,
        tuner: Optional[str] = 'greedy',
        validation_split: Optional[float] = 0.1,
        epochs: Optional[int] = 10,
        seed: Optional[int] = None
    ):
        """
        Imputer uses `AutoKeras` 

        Args:
            max_trials (Optional[int] = 10): maximum number of trials for model selection
            tuner (Optional[str] = 'greedy'): autokeras hyperparameter tuning strategy
            validiation_split (Optional[float] = 0.1): validation split for autokeras fit
            epochs (Optional[int] = 10): number of epochs for autokeras fit
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        super().__init__(
            seed=seed
        )
        self.max_trials = max_trials
        self.epochs = epochs
        self.validation_split = validation_split
        self.tuner = tuner
        self._predictors = {}

    def get_best_hyperparameters(self):
        pass

    def fit(self, data: pd.DataFrame, target_columns: List[str]) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns)

        # cast categorical columns to strings fixes problems where categories are integer values and treated as regression task
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for target in self._target_columns:
            # NOTE: We use the whole available data.
            # If there are missing values in predictor columns, they getting imputed (marked) beforehand to use them for fitting.
            feature_cols = [c for c in self._categorical_columns + \
                self._numerical_columns if c != target]

            if target in self._numerical_columns:
                akm = StructuredDataRegressor
            elif target in self._categorical_columns:
                akm = StructuredDataClassifier

            self._predictors[target] = akm(
                    column_names = feature_cols,
                    overwrite=True,
                    max_trials=self.max_trials,
                    tuner=self.tuner
                )
    
            missing_mask = data[target].isna()
            self._predictors[target].fit(data.loc[~missing_mask, feature_cols], data.loc[~missing_mask, target])

        self._fitted = True

        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        super().transform(data=data)

        imputed_mask = data[self._target_columns].isna()

        # save the original dtypes because ..
        dtypes = data.dtypes

        # ... dtypes of data need to be same as for fitting
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for target in self._target_columns:
            feature_cols = [c for c in self._categorical_columns + \
                self._numerical_columns if c != target]
            missing_mask = data[target].isna()
            amount_missing_in_columns = missing_mask.sum()

            if amount_missing_in_columns > 0:
                data.loc[missing_mask, target] = self._predictors[target].predict(data.loc[missing_mask,feature_cols])
                logger.debug(f'Imputed {amount_missing_in_columns} values in column {target}')

        self._restore_dtype(data, dtypes)

        return data, imputed_mask


