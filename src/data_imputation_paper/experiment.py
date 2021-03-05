import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
from jenga.tasks.openml import OpenMLTask

from .evaluation import (
    EvaluationResult,
    MultipleColumnsAllMissingEvaluator,
    MultipleColumnsEvaluator,
    SingleColumnAllMissingEvaluator,
    SingleColumnEvaluator
)
from .imputation._base import BaseImputer
from .imputation.utils import set_seed

logger = logging.getLogger()


strategy_to_EvaluatorClass = {
    "single_single": SingleColumnEvaluator,
    "multiple_multiple": MultipleColumnsEvaluator,
    "single_all": SingleColumnAllMissingEvaluator,
    "multiple_all": MultipleColumnsAllMissingEvaluator
}


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
        seed: int = 42
    ):

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

        valid_strategies = strategy_to_EvaluatorClass.keys()
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

    def run(self):
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
                            evaluator = strategy_to_EvaluatorClass[strategy](
                                task=task,
                                missing_fraction=missing_fraction,
                                missing_type=missing_type,
                                target_column=task.train_data.columns.tolist()[0] if "single" in strategy else task.train_data.columns.tolist()[:2],  # TODO We need to find a way to sample here!
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
