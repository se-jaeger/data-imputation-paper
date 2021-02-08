import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ._base import BaseImputer, SklearnBaseImputer

logger = logging.getLogger()


class ModeImputer(SklearnBaseImputer):

    def __init__(self, seed: Optional[int] = None):
        """
        Imputer that fills missing values with the column's mean (for numerical columns) or most frequent (for categorical columns) value.

        Args:
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        # BaseImputer bootstraps the object
        BaseImputer.__init__(self, seed=seed)

        self._predictors: Dict[str, float] = {}

    def fit(self, data: pd.DataFrame, target_columns: List[str]) -> BaseImputer:

        # BaseImputer does some error checking and bootstrap
        BaseImputer.fit(self, data, target_columns)

        for column in self._target_columns:
            if column in self._categorical_columns:
                self._predictors[column] = data[column].mode()[0]  # It's possible that there are more than one values most frequent

            elif column in self._numerical_columns:
                self._predictors[column] = data[column].mean(axis=0)

        self._fitted = True

        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        BaseImputer.transform(self, data=data)

        imputed_mask = data[self._target_columns].isna()

        # save the original dtypes because ..
        dtypes = data.dtypes

        # ... dtypes of data need to be same as for fitting
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        for column in self._target_columns:
            missing_mask = data[column].isna()
            amount_missing_in_columns = missing_mask.sum()

            if amount_missing_in_columns > 0:
                data.loc[missing_mask, column] = self._predictors[column]

                logger.debug(f'Imputed {amount_missing_in_columns} values in column {column}')

        self._restore_dtype(data, dtypes)

        return data, imputed_mask

    def get_best_hyperparameters(self) -> dict:
        return {}
