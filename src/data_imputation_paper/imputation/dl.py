from typing import Optional, List, Tuple

from autokeras import StructuredDataClassifier, StructuredDataRegressor

import logging
import pandas as pd

from ._base import BaseImputer


logger = logging.getLogger()


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

        # TODO..
        pass

    def fit(self, data: pd.DataFrame, target_columns: List[str]) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns)

        # cast categorical columns to strings fixes problems where categories are integer values and treated as regression task
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for target in self._target_columns:

            feature_cols = [c for c in self._categorical_columns + self._numerical_columns if c != target]

            if target in self._numerical_columns:
                StructuredDataModel = StructuredDataRegressor

            elif target in self._categorical_columns:
                StructuredDataModel = StructuredDataClassifier

            self._predictors[target] = StructuredDataModel(
                    column_names=feature_cols,
                    overwrite=True,
                    max_trials=self.max_trials,
                    tuner=self.tuner
                )

            missing_mask = data[target].isna()
            self._predictors[target].fit(
                x=data.loc[~missing_mask, feature_cols],
                y=data.loc[~missing_mask, target],
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

        for target in self._target_columns:
            feature_cols = [c for c in self._categorical_columns + self._numerical_columns if c != target]
            missing_mask = data[target].isna()
            amount_missing_in_columns = missing_mask.sum()

            if amount_missing_in_columns > 0:
                data.loc[missing_mask, target] = self._predictors[target].predict(data.loc[missing_mask, feature_cols])
                logger.debug(f'Imputed {amount_missing_in_columns} values in column {target}')

        self._restore_dtype(data, dtypes)

        return data, imputed_mask
