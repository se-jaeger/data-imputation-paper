import logging
from typing import Dict, List, Tuple

import pandas as pd

from ._base import BaseImputer, SklearnBaseImputer

logger = logging.getLogger()


class ModeImputer(SklearnBaseImputer):

    def __init__(self, grid_imputer_arguments: dict = {}):

        # BaseImputer bootstraps the object
        BaseImputer.__init__(self)

        self._predictors: Dict[str, float] = {}

    def fit(self, data: pd.DataFrame, target_columns: List[str], refit: bool = False) -> BaseImputer:

        # BaseImputer does some error checking and bootstrap
        BaseImputer.fit(self, data, target_columns, refit)

        for column in self._target_columns:
            if column in self._categorical_columns:
                self._predictors[column] = data[column].mode()[0]  # It's possible that there are more than one values most frequent

            elif column in self._numerical_columns:
                self._predictors[column] = data[column].mean(axis=0)

        self._fitted = True

        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

        # save the original dtypes because ..
        dtypes = data.dtypes

        # ... dtypes of data need to be same as for fitting
        data = self._categorical_columns_to_string(data.copy())  # We don't want to change the input dataframe -> copy it

        imputed_mask = data.isna().any(axis=1)

        for column in self._target_columns:
            missing_mask = data[column].isna()
            amount_missing_in_columns = missing_mask.sum()

            if amount_missing_in_columns > 0:
                data.loc[missing_mask, column] = self._predictors[column]

                logger.debug(f'Imputed {amount_missing_in_columns} values in column {column}')

        self._restore_dtype(data, dtypes)

        return data, imputed_mask
