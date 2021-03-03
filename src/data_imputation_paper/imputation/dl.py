import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from autokeras import StructuredDataClassifier, StructuredDataRegressor
from tensorflow.keras import Model

from ._base import BaseImputer

logger = logging.getLogger()


class AutoKerasImputer(BaseImputer):

    def __init__(
        self,
        max_trials: Optional[int] = 10,
        tuner: Optional[str] = None,
        validation_split: Optional[float] = 0.2,
        epochs: Optional[int] = 10,
        seed: Optional[int] = None
    ):
        """
        Deep Learning-learning based imputation mehtod. It uses AutoKeras to find good architecture/hyperparameters.

        Args:
            max_trials (Optional[int], optional): maximum number of trials for model selection. Defaults to 10.
            tuner (Optional[str], optional): AutoKeras hyperparameter tuning strategy. Defaults to None.
            validation_split (Optional[float], optional): validation split for AutoKeras fit. Defaults to 0.2.
            epochs (Optional[int], optional): number of epochs for AutoKeras fit. Defaults to 10.
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        super().__init__(
            seed=seed
        )

        self.max_trials = max_trials
        self.epochs = epochs
        self.validation_split = validation_split
        self.tuner = tuner
        self._predictors: Dict[str, Model] = {}

    def get_best_hyperparameters(self):

        super().get_best_hyperparameters()

        return {
            column: self._predictors[column].tuner.get_best_hyperparameters()[0].values
            for column in self._predictors.keys()
        }

    def fit(self, data: pd.DataFrame, target_columns: List[str]) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns)

        # cast categorical columns to strings fixes problems where categories are integer values and treated as regression task
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for target_column in self._target_columns:

            missing_mask = data[target_column].isna()
            feature_cols = [c for c in self._categorical_columns + self._numerical_columns if c != target_column]

            if target_column in self._numerical_columns:
                StructuredDataModelSearch = StructuredDataRegressor

            elif target_column in self._categorical_columns:
                StructuredDataModelSearch = StructuredDataClassifier

            self._predictors[target_column] = StructuredDataModelSearch(
                column_names=feature_cols,
                overwrite=True,
                max_trials=self.max_trials,
                tuner=self.tuner,
                directory="../models"
            )

            self._predictors[target_column].fit(
                x=data.loc[~missing_mask, feature_cols],
                y=data.loc[~missing_mask, target_column],
                epochs=self.epochs
            )

        self._fitted = True

        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        super().transform(data=data)

        imputed_mask = data[self._target_columns].isna()

        # save the original dtypes because ..
        dtypes = data.dtypes

        # ... dtypes of data need to be same as for fitting
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for target_column in self._target_columns:
            feature_cols = [c for c in self._categorical_columns + self._numerical_columns if c != target_column]
            missing_mask = data[target_column].isna()
            amount_missing_in_columns = missing_mask.sum()

            if amount_missing_in_columns > 0:
                data.loc[missing_mask, target_column] = self._predictors[target_column].predict(data.loc[missing_mask, feature_cols])[:, 0]
                logger.debug(f'Imputed {amount_missing_in_columns} values in column {target_column}')

        self._restore_dtype(data, dtypes)

        return data, imputed_mask
