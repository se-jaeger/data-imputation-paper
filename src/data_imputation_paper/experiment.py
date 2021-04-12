import json
import logging
import os
import random
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from jenga.tasks.openml import OpenMLTask

from .imputation._base import BaseImputer
from .imputation.utils import set_seed

logger = logging.getLogger()


class Experiment(object):

    def __init__(
        self,
        task_id_class_tuples: List[Tuple[int, Callable[..., OpenMLTask]]],
        missing_fractions: List[float],
        missing_types: List[str],
        strategies: List[str],
        imputer_class: BaseImputer,
        imputer_arguments: dict,
        num_repetitions: int,
        base_path: str = "results",
        timestamp: Optional[str] = None,
        fully_observed: bool = True,
        seed: int = 42
    ):

        if fully_observed:
            from .evaluation import (
                EvaluationResult,
                MultipleColumnsAllMissingEvaluator,
                MultipleColumnsEvaluator,
                SingleColumnAllMissingEvaluator,
                SingleColumnEvaluator
            )
        else:
            from .evaluation_corrupted import (
                EvaluationResult,
                MultipleColumnsAllMissingEvaluator,
                MultipleColumnsEvaluator,
                SingleColumnAllMissingEvaluator,
                SingleColumnEvaluator
            )

        self.strategy_to_EvaluatorClass = {
            "single_single": SingleColumnEvaluator,
            "multiple_multiple": MultipleColumnsEvaluator,
            "single_all": SingleColumnAllMissingEvaluator,
            "multiple_all": MultipleColumnsAllMissingEvaluator
        }

        self._task_id_class_tuples = task_id_class_tuples
        self._missing_fractions = missing_fractions
        self._missing_types = missing_types
        self._strategies = strategies
        self._imputer_class = imputer_class
        self._imputer_arguments = imputer_arguments
        self._num_repetitions = num_repetitions
        self._timestamp = timestamp
        self._seed = seed
        self._result: Dict[int, Dict[str, Dict[float, Dict[str, EvaluationResult]]]] = dict()

        valid_strategies = self.strategy_to_EvaluatorClass.keys()
        for strategy in self._strategies:
            if strategy not in valid_strategies:
                raise Exception(f"'{strategy}' is not a valid strategy. Need to be in {', '.join(valid_strategies)}")

        self._base_path = Path(base_path)

        if self._timestamp is None:
            self._timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")

        imputer_class_name = str(self._imputer_class).split("'")[-2].split(".")[-1]
        self._base_path = self._base_path / self._timestamp / imputer_class_name

        # Because we set determinism here, supres downstream determinism mechanisms
        if self._seed:
            set_seed(self._seed)
            self._imputer_arguments.pop("seed", None)

    def run(self, target_column):
        for task_id, task_class in self._task_id_class_tuples:
            self._result[task_id] = {}

            task = task_class(openml_id=task_id)

            for missing_type in self._missing_types:
                self._result[task_id][missing_type] = {}

                for missing_fraction in self._missing_fractions:
                    self._result[task_id][missing_type][missing_fraction] = {}

                    for strategy in self._strategies:

                        experiment_path = self._base_path / f"{task_id}" / missing_type / f"{missing_fraction}" / f"{strategy}"

                        try:
                            evaluator = self.strategy_to_EvaluatorClass[strategy](
                                task=task,
                                missing_fraction=missing_fraction,
                                missing_type=missing_type,
                                target_column=target_column,
                                imputer_class=self._imputer_class,
                                imputer_args=self._imputer_arguments,
                                path=experiment_path,
                                seed=None
                            )
                            evaluator.evaluate(self._num_repetitions)
                            result = evaluator._result

                        except Exception:
                            error = traceback.format_exc()
                            experiment_path.mkdir(parents=True, exist_ok=True)
                            Path(experiment_path / "error.txt").write_text(str(error))
                            logger.exception(f"Tried to run - missing type: {missing_type} - missing fraction: {missing_fraction}")
                            result = error

                        self._result[task_id][missing_type][missing_fraction][strategy] = result

        joblib.dump(self._result, Path(self._base_path / f"{task_id}" / "result.joblib"))
        Path(self._base_path / f"{task_id}" / "evaluation_parameters.json").write_text(
            json.dumps(
                {
                    "missing_types": self._missing_types,
                    "missing_fractions": self._missing_fractions,
                    "strategy": self._strategies
                }
            )
        )

        logger.info(f"Experiment Finished! - Results are at: {self._base_path.parent}")


def _recursive_split(path):
    """
    Recursively splits a path into its components.

    Returns:
        tuple
    """
    rest, tail = os.path.split(path)
    if rest in ('', os.path.sep):
        return tail,
    return _recursive_split(rest) + (tail,)


def read_experiment(path):
    """
    Discovers CSV files an experiment produced and construct columns
    for the experiment's conditions from the sub-directory structure.

    Args:
        path: path to the experiment's results.

    Returns:
        pd.DataFrame
    """
    objects = list(path.rglob('*.csv'))
    data = []
    path_split = _recursive_split(path)

    for obj in objects:
        obj_path_split = _recursive_split(obj)
        if len(obj_path_split) - len(path_split) > 7:
            raise Exception("Path depth too long! Provide path to actual experiment or one of its sub-directories.")
        data.append(obj_path_split)

    df = pd.DataFrame(data=data)

    columns = ["experiment", "imputer", "task", "missing_type", "missing_fraction", "strategy", "file_or_dir", "detail_file"]
    auto_columns = []
    for i in range(df.shape[1] - len(columns)):
        auto_columns.append(f"col{i}")
    df.columns = auto_columns + columns
    df.drop(auto_columns, axis=1, inplace=True)

    df["path"] = objects
    df["detail_file"] = df["detail_file"].fillna("")

    return df.reset_index(drop=True)


def _read_prefixed_csv_files(df_experiment, file_prefix, read_details):
    col_pattern = f"({file_prefix}_)(\\S*)(.csv)"
    dfs = []
    if read_details:
        file_col = "detail_file"
    else:
        file_col = "file_or_dir"
    # TODO this loop is pretty slow
    for row in df_experiment[df_experiment[file_col].str.startswith(file_prefix)].iterrows():
        df_new = pd.read_csv(row[1]["path"])
        df_new.rename({"Unnamed: 0": "metric"}, inplace=True, axis=1)
        df_new["experiment"] = row[1]["experiment"]
        df_new["imputer"] = row[1]["imputer"]
        df_new["task"] = row[1]["task"]
        df_new["missing_type"] = row[1]["missing_type"]
        df_new["missing_fraction"] = row[1]["missing_fraction"]
        df_new["strategy"] = row[1]["strategy"]
        if read_details:
            df_new["column"] = row[1]["file_or_dir"]
        else:
            # column name contained in file names
            df_new["column"] = re.findall(col_pattern, row[1][file_col])[0][1]
        df_new["result_type"] = file_prefix
        dfs.append(df_new)
    return pd.concat(dfs, ignore_index=True)


def read_csv_files(df_experiment, read_details=True):
    """
    Reads data from the CSV files which were produced by an experiment, i.e. the results.

    Args:
        df_experiment: pd.DataFrame containing the conditions as well as names/path of the CSV files of an experiment.

    Returns:
        pd.DataFrame with all experiment conditions and (aggregated) scores
    """
    if read_details:
        result_types = [
            "impute_performance",
            "downstream_performance"
        ]
    else:
        result_types = [
            "impute_performance_std",
            "impute_performance_mean",
            "downstream_performance_std",
            "downstream_performance_mean"
        ]
    df_experiment = pd.concat(
        [_read_prefixed_csv_files(df_experiment, rt, read_details) for rt in result_types],
        ignore_index=True
    )
    df_experiment["missing_fraction"] = pd.to_numeric(df_experiment["missing_fraction"])

    ordered_columns = [
        "experiment", "imputer", "task", "missing_type", "missing_fraction", "strategy", "column",
        "result_type", "metric", "train", "test", "baseline", "corrupted", "imputed"
    ]
    assert len(ordered_columns) == df_experiment.shape[1]
    df_experiment = df_experiment[ordered_columns]

    return df_experiment
