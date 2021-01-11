import math
from typing import List

import pandas as pd
from jenga.corruptions.generic import MissingValues
from jenga.tasks.openml import OpenMLTask
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

from .imputation._base import BaseImputer


class EvaluationError(Exception):
    """Exception raised for errors in Evaluation classes"""
    pass


class EvaluationResult(object):

    def __init__(self, task: OpenMLTask, missing_value: MissingValues):

        self._task = task
        self._missing_value = missing_value
        self._target_column = self._missing_value.column
        self._finalized = False
        self.results: List[pd.DataFrame] = []
        self.repetitions = 0

        self._set_imputation_task_type()

    def append(self, train_data_imputed: pd.DataFrame, test_data_imputed: pd.DataFrame):

        if self._finalized:
            raise EvaluationError("Evaluation already finalized")

        self._update_results(
            train=self._task.train_data[self._target_column],
            train_imputed=train_data_imputed[self._target_column],
            test=self._task.test_data[self._target_column],
            test_imputed=test_data_imputed[self._target_column],
            imputation_type=self._imputation_task_type
        )

        self.repetitions += 1

    def finalize(self):

        if self._finalized:
            raise EvaluationError("Evaluation already finalized")

        collected_results = pd.concat(self.results)
        indices = collected_results.index.unique()

        self.result = pd.DataFrame(
            [
                collected_results.loc[metric].mean() for metric in indices
            ],
            index=indices
        )

        self._finalized = True

        return self

    def _set_imputation_task_type(self):
        if pd.api.types.is_numeric_dtype(self._task.train_data[self._target_column]):
            self._imputation_task_type = "regression"

        elif pd.api.types.is_categorical_dtype(self._task.train_data[self._target_column]):
            num_classes = len(self._task.train_data[self._target_column].dtype.categories)

            if num_classes == 2:
                self._imputation_task_type = "binary_classification"

            elif num_classes > 2:
                self._imputation_task_type = "multiclass_classification"

            else:
                raise EvaluationError(f"Found categorical imputation with {num_classes} categories")

        else:
            raise EvaluationError(f"datatype of column '{self._target_column}' recognized")

    # TODO: reduce code...
    def _update_results(
        self,
        train: pd.Series,
        train_imputed: pd.Series,
        test: pd.Series,
        test_imputed: pd.Series,
        imputation_type: str
    ):

        if self._imputation_task_type == "regression":
            self.results.append(
                pd.DataFrame(
                    {
                        "train": {
                            "MAE": mean_absolute_error(train, train_imputed),
                            "MSE": mean_squared_error(train, train_imputed),
                            "RMSE": math.sqrt(mean_squared_error(train, train_imputed))
                        },
                        "test": {
                            "MAE": mean_absolute_error(test, test_imputed),
                            "MSE": mean_squared_error(test, test_imputed),
                            "RMSE": math.sqrt(mean_squared_error(test, test_imputed))
                        }
                    }
                )
            )

        elif "classification" in self._imputation_task_type:
            self.results.append(
                pd.DataFrame(
                    {
                        "train": {
                            "F1_micro": f1_score(train, train_imputed, average="micro"),
                            "F1_macro": f1_score(train, train_imputed, average="macro"),
                            "F1_weighted": f1_score(train, train_imputed, average="weighted"),
                        },
                        "test": {
                            "F1_micro": f1_score(test, test_imputed, average="micro"),
                            "F1_macro": f1_score(test, test_imputed, average="macro"),
                            "F1_weighted": f1_score(test, test_imputed, average="weighted"),
                        }
                    }
                )
            )


class Evaluator(object):

    def __init__(self, task: OpenMLTask, missing_value: MissingValues, imputer: BaseImputer):
        self._task = task
        self._missing_value = missing_value
        self._imputer = imputer
        self._target_column = self._missing_value.column

    def evaluate(self, num_repetitions: int) -> EvaluationResult:

        result = EvaluationResult(self._task, self._missing_value)

        for _ in tqdm(range(num_repetitions)):
            missing_train = self._missing_value.transform(self._task.train_data)
            missing_test = self._missing_value.transform(self._task.test_data)

            self._imputer.fit(self._task.train_data, [self._target_column], refit=True)

            train_imputed, train_imputed_mask = self._imputer.transform(missing_train)
            test_imputed, test_imputed_mask = self._imputer.transform(missing_test)

            result.append(train_imputed, test_imputed)

        return result.finalize()
