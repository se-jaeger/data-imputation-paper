import json
import math
import time
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
        self.elapsed_train_times: List[float] = []
        self.best_hyperparameters: List[Dict[str, List[dict]]] = []
        self.repetitions = 0

        if self._task._task_type == BINARY_CLASSIFICATION or self._task._task_type == MULTI_CLASS_CLASSIFICATION:
            self._baseline_metric = ("F1_micro", "F1_macro", "F1_weighted")

        elif self._task._task_type == REGRESSION:
            self._baseline_metric = ("MAE", "MSE", "RMSE")

        self._baseline_performance = self._task.get_baseline_performance()

        self._set_imputation_task_type()

    def append(
        self,
        target_column: str,
        train_data_imputed: pd.DataFrame,
        test_data_imputed: pd.DataFrame,
        test_data_corrupted: pd.DataFrame,
        train_imputed_mask: pd.Series,
        test_imputed_mask: pd.Series,
        elapsed_time: float,
        best_hyperparameters: Dict[str, List[dict]]
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

        predictions_on_corrupted = self._task._baseline_model.predict(test_data_corrupted)
        score_on_corrupted = self._task.score_on_test_data(predictions_on_corrupted)

        predictions_on_imputed = self._task._baseline_model.predict(test_data_imputed)
        score_on_imputed = self._task.score_on_test_data(predictions_on_imputed)

        self.downstream_performances.append(
            pd.DataFrame(
                {
                    "baseline": {
                        self._baseline_metric[0]: self._baseline_performance[0],
                        self._baseline_metric[1]: self._baseline_performance[1],
                        self._baseline_metric[2]: self._baseline_performance[2]
                    },
                    "corrupted": {
                        self._baseline_metric[0]: score_on_corrupted[0],
                        self._baseline_metric[1]: score_on_corrupted[1],
                        self._baseline_metric[2]: score_on_corrupted[2]
                    },
                    "imputed": {
                        self._baseline_metric[0]: score_on_imputed[0],
                        self._baseline_metric[1]: score_on_imputed[1],
                        self._baseline_metric[2]: score_on_imputed[2]
                    }
                }
            )
        )

        self.elapsed_train_times.append(elapsed_time)
        self.best_hyperparameters.append(best_hyperparameters)

        self.repetitions += 1

    def finalize(self):

        if self._finalized:
            raise EvaluationError("Evaluation already finalized")

        results_reduced = []
        for all_list in [self.results, self.downstream_performances]:
            collected_results = pd.concat(all_list)
            metrics = collected_results.index.unique()

            results_reduced.append(
                pd.DataFrame(
                    [
                        collected_results.loc[metric].mean() for metric in metrics
                    ],
                    index=metrics
                )
            )
        self.result, self.downstream_performance = results_reduced

        self.elapsed_train_time = sum(self.elapsed_train_times) / len(self.elapsed_train_times)

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
            print(f"Target Column: {key} - Necessary train time in seconds: {round(value.elapsed_train_time, 4)}")
            print(value.result)
            print()
            print(value.downstream_performance)
            print("\n")

    def evaluate(self, num_repetitions: int):

        result = {}

        for target_column in [missing_value.column for missing_value in self._missing_values]:  # TODO hier..

            result_temp = EvaluationResult(self._task, target_column)

            for _ in range(num_repetitions):
                missing_train, missing_test = self._apply_missing_values(self._task, self._missing_values)

                imputer = self._imputer_class(**self._imputer_arguments)

                start_time = time.time()
                imputer.fit(self._task.train_data.copy(), [target_column])
                elapsed_time = time.time() - start_time

                train_imputed, train_imputed_mask = imputer.transform(missing_train)
                test_imputed, test_imputed_mask = imputer.transform(missing_test)

                # NOTE: masks are DataFrames => append expects Series
                result_temp.append(
                    target_column=target_column,
                    train_data_imputed=train_imputed,
                    test_data_imputed=test_imputed,
                    test_data_corrupted=missing_test,
                    train_imputed_mask=train_imputed_mask[target_column],
                    test_imputed_mask=test_imputed_mask[target_column],
                    elapsed_time=elapsed_time,
                    best_hyperparameters=imputer.get_best_hyperparameters()
                )

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
                Path(self._path / f"best_hyperparameters_{column}.json").write_text(json.dumps(self._result[column].best_hyperparameters))
                Path(self._path / f"elapsed_train_time_{column}.json").write_text(json.dumps(self._result[column].elapsed_train_times))

                results_path = self._path / column
                results_path.mkdir(parents=True, exist_ok=True)

                for index, (impute_data_frame, performance_data_frame, best_hyperparameters, elapsed_train_time) in enumerate(
                    zip(
                        self._result[column].results,
                        self._result[column].downstream_performances,
                        self._result[column].best_hyperparameters,
                        self._result[column].elapsed_train_times
                    )
                ):

                    impute_data_frame.to_csv(results_path / f"impute_performance_rep_{index}.csv")
                    performance_data_frame.to_csv(results_path / f"downstream_performance_rep_{index}.csv")
                    Path(results_path / f"elapsed_train_time_rep_{index}.json").write_text(json.dumps(elapsed_train_time))
                    Path(results_path / f"best_hyperparameters_rep_{index}.json").write_text(json.dumps(best_hyperparameters))
