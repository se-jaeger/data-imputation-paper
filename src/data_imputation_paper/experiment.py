import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
from jenga.corruptions.generic import MissingValues
from jenga.tasks.openml import OpenMLTask

from .evaluation import EvaluationResult, Evaluator
from .imputation._base import BaseImputer
from .imputation.utils import set_seed

logger = logging.getLogger()

# TODO: next steps:
# - Plotting results..


class Experiment(object):

    def __init__(
        self,
        task_ids: List[str],
        missing_fractions: List[float],
        missing_types: List[str],
        imputer_class: BaseImputer,
        imputer_arguments: dict,
        num_repetitions: int,
        base_path: str = "results",
        seed: int = 42
    ):

        self._task_ids = task_ids
        self._missing_fractions = missing_fractions
        self._missing_types = missing_types
        self._imputer_class = imputer_class
        self._imputer_arguments = imputer_arguments
        self._num_repetitions = num_repetitions
        self._seed = seed
        self._result: Dict[int, Dict[str, Dict[float, Dict[str, EvaluationResult]]]] = dict()

        self._base_path = Path(base_path)

        imputer_class_name = str(self._imputer_class).split("'")[-2].split(".")[-1]
        self._base_path = self._base_path / datetime.now().strftime("%Y-%m-%d_%H:%M") / imputer_class_name

        # Because we set determinism here, supres downstream determinism mechanisms
        if self._seed:
            set_seed(self._seed)
            self._imputer_arguments.pop("seed", None)

    def run(self):
        for task_id in self._task_ids:
            self._result[task_id] = {}

            task = OpenMLTask(seed=self._seed, openml_id=task_id)

            for missing_type in self._missing_types:
                self._result[task_id][missing_type] = {}

                for missing_fraction in self._missing_fractions:
                    # TODO: We probably want to change this
                    missing_values = [
                        MissingValues(column=column, fraction=missing_fraction, missingness=missing_type)
                        for column in task.train_data.columns
                    ]

                    experiment_path = self._base_path / f"{task_id}" / missing_type / f"{missing_fraction}"
                    evaluator = Evaluator(task, missing_values, self._imputer_class, self._imputer_arguments, experiment_path, seed=None)
                    evaluator.evaluate(self._num_repetitions)

                    self._result[task_id][missing_type][missing_fraction] = evaluator._result

        joblib.dump(self._result, self._base_path.parent / "result.joblib")
        logger.info(f"Experiment Finished! - Results are at: {self._base_path.parent}")
