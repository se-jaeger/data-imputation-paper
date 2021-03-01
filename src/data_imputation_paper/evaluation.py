import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from jenga.corruptions.generic import MissingValues
from jenga.tasks.openml import OpenMLTask
from jenga.utils import BINARY_CLASSIFICATION, MULTI_CLASS_CLASSIFICATION, REGRESSION
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

from .imputation._base import BaseImputer
from .imputation.utils import set_seed


class EvaluationError(Exception):
    """Exception raised for errors in Evaluation classes"""
    pass


class EvaluationResult(object):

    def __init__(self, task: OpenMLTask, target_column: str):

        self._task = task
        self._target_column = target_column
        self._finalized = False
        self.results: List[pd.DataFrame] = []
        self.downstream_performances: List[pd.DataFrame] = []
        self.repetitions = 0

        if self._task._task_type == BINARY_CLASSIFICATION:
            self._baseline_metric = "ROC/AUC"
        elif self._task._task_type == MULTI_CLASS_CLASSIFICATION:
            self._baseline_metric = "F1"
        elif self._task._task_type == REGRESSION:
            self._baseline_metric = "MSE"

        self._baseline_performance = self._task.get_baseline_performance()

        self._set_imputation_task_type()

    def append(
        self,
        target_column: str,
        train_data_imputed: pd.DataFrame,
        test_data_imputed: pd.DataFrame,
        train_imputed_mask: pd.Series,
        test_imputed_mask: pd.Series
    ):

        if self._finalized:
            raise EvaluationError("Evaluation already finalized")

        self._update_results(
            train=self._task.train_data.loc[train_imputed_mask, target_column],
            train_imputed=train_data_imputed.loc[train_imputed_mask, target_column],
            test=self._task.test_data.loc[test_imputed_mask, target_column],
            test_imputed=test_data_imputed.loc[test_imputed_mask, target_column],
            imputation_type=self._imputation_task_type
        )

        model = self._task.fit_baseline_model(train_data_imputed, self._task.train_labels)

        if self._task._task_type == BINARY_CLASSIFICATION:
            predictions = model.predict_proba(test_data_imputed)
        else:
            predictions = model.predict(test_data_imputed)

        score = self._task.score_on_test_data(predictions)
        self.downstream_performances.append(pd.DataFrame([[score, self._baseline_metric]], columns=["score", "metric"]))

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

        collected_scores = pd.concat(self.downstream_performances)
        self.downstream_performance = pd.DataFrame(
            {
                "score": {
                    "baseline": self._baseline_performance,
                    "imputed": collected_scores["score"].mean()
                },
                "metric": {
                    "baseline": self._baseline_metric,
                    "imputed": self._baseline_metric
                }
            }
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
            raise EvaluationError(f"datatype of column '{self._target_column}' not recognized")

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

    def __init__(
        self,
        task: OpenMLTask,
        missing_values: List[MissingValues],
        imputer_class: Callable[..., BaseImputer],
        imputer_args: dict,
        path: Optional[Path] = None,
        seed: Optional[int] = 42
    ):

        self._task = task
        self._missing_values = missing_values
        self._imputer_class = imputer_class
        self._imputer_arguments = imputer_args
        self._path = path
        self._result: Optional[Dict[str, EvaluationResult]] = None
        self._best_hyperparameters: Dict[str, List[dict]] = dict()
        self._seed = seed

        # fit task's baseline model and get performance
        self._task.fit_baseline_model()

        # Because we set determinism here, supres downstream determinism mechanisms
        if self._seed:
            set_seed(self._seed)
            self._imputer_arguments.pop("seed", None)

    @staticmethod
    def report_results(result_dictionary: Dict[str, EvaluationResult]) -> None:
        target_columns = list(result_dictionary.keys())

        print(f"Evaluation result contains {len(target_columns)} target columns: {', '.join(target_columns)}")
        print("All are in a round-robin fashion imputed and performances are as follows:\n")

        for key, value in result_dictionary.items():
            print(f"Target Column: {key}")
            print(value.result)
            print()
            print(value.downstream_performance)
            print("\n")

    def evaluate(self, num_repetitions: int):

        result = {}

        for target_column in [missing_value.column for missing_value in self._missing_values]:

            result_temp = EvaluationResult(self._task, target_column)
            self._best_hyperparameters[target_column] = []

            for _ in range(num_repetitions):
                missing_train, missing_test = self._apply_missing_values(self._task, self._missing_values)

                imputer = self._imputer_class(**self._imputer_arguments)
                imputer.fit(missing_train, [target_column])

                train_imputed, train_imputed_mask = imputer.transform(missing_train)
                test_imputed, test_imputed_mask = imputer.transform(missing_test)

                # NOTE: masks are DataFrames => append expects Series
                result_temp.append(target_column, train_imputed, test_imputed, train_imputed_mask[target_column], test_imputed_mask[target_column])
                self._best_hyperparameters[target_column].append(imputer.get_best_hyperparameters())

            result[target_column] = result_temp.finalize()

        self._result = result
        self._save_results()

        return self

    def report(self) -> None:
        if self._result is None:
            raise EvaluationError("Not evaluated yet. Call 'evaluate' first!")

        else:
            self.report_results(self._result)

    def _apply_missing_values(self, task: OpenMLTask, missing_values: List[MissingValues]) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train_data = task.train_data.copy()
        test_data = task.test_data.copy()

        for missing_value in missing_values:
            train_data = missing_value.transform(train_data)
            test_data = missing_value.transform(test_data)

        return (train_data, test_data)

    def _save_results(self):
        if self._path is not None:
            self._path.mkdir(parents=True, exist_ok=True)

            for column in self._result.keys():
                self._result[column].result.to_csv(self._path / f"impute_performance_{column}.csv")
                self._result[column].downstream_performance.to_csv(self._path / f"downstream_performance_{column}.csv")

                results_path = self._path / column
                results_path.mkdir(parents=True, exist_ok=True)

                for index, (impute_data_frame, performance_data_frame) in enumerate(
                    zip(self._result[column].results, self._result[column].downstream_performances)
                ):

                    impute_data_frame.to_csv(results_path / f"impute_performance_rep_{index}.csv")
                    performance_data_frame.to_csv(results_path / f"downstream_performance_rep_{index}.csv")
                    (results_path / f"best_hyperparameters_rep_{index}.json").write_text(json.dumps(self._best_hyperparameters[column]))
