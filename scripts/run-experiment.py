
from pathlib import Path
from typing import List, Tuple

import typer
from jenga.tasks.openml import (
    OpenMLBinaryClassificationTask,
    OpenMLMultiClassClassificationTask,
    OpenMLRegressionTask,
    OpenMLTask
)

from data_imputation_paper.experiment import Experiment
from data_imputation_paper.imputation._base import BaseImputer
from data_imputation_paper.imputation.dl import AutoKerasImputer
from data_imputation_paper.imputation.generative import GAINImputer
from data_imputation_paper.imputation.ml import ForestImputer, KNNImputer
from data_imputation_paper.imputation.simple import ModeImputer

IMPUTER_CLASS = {
    "mode": ModeImputer,
    "knn": KNNImputer,
    "forest": ForestImputer,
    "dl": AutoKerasImputer,
    "gain": GAINImputer,
    "vae": None  # TODO
}

IMPUTER_NAME = {
    "mode": "ModeImputer",
    "knn": "KNNImputer",
    "forest": "ForestImputer",
    "dl": "AutoKerasImputer",
    "gain": "GAINImputer",
    "vae": None  # TODO
}

# TODO...
IMPUTER_ARGUMENTS = {
    "mode": {},  # NOTE: there are no arguments here..
    "knn": {
        "hyperparameter_grid_categorical_imputer": {
            "n_neighbors": [3, 5]
        },
        "hyperparameter_grid_numerical_imputer": {
            "n_neighbors": [3, 5]
        }
    },
    "forest": {
        "hyperparameter_grid_categorical_imputer": {
            "n_estimators": [50, 100]
        },
        "hyperparameter_grid_numerical_imputer": {
            "n_estimators": [50, 100]
        }
    },
    "dl": {
        "max_trials": 10,
        "tuner": None,
        "validation_split": 0.2,
        "epochs": 10
    },
    "gain": {
        "hyperparameter_grid": {
            "gain": {
                "alpha": [100],
                "hint_rate": [0.9],
                "noise": [0.01]
            },
            "training": {
                "batch_size": [48],
                "epochs": [10]
            },
            "generator": {
                "learning_rate": [0.0005],
                "beta_1": [0.9],
                "beta_2": [0.999],
                "epsilon": [1e-7],
                "amsgrad": [False]
            },
            "discriminator": {
                "learning_rate": [0.00005],
                "beta_1": [0.9],
                "beta_2": [0.999],
                "epsilon": [1e-7],
                "amsgrad": [False]
            }
        }
    },
    "vae": {
        # TODO
    }
}

binary_path = Path("../data/raw/binary.txt")
multi_path = Path("../data/raw/multi.txt")
regression_path = Path("../data/raw/regression.txt")

BINARY_TASK_IDS = [int(x) for x in binary_path.read_text().split(",")]
MULTI_TASK_IDS = [int(x) for x in multi_path.read_text().split(",")]
REGRESSION_TASK_IDS = [int(x) for x in regression_path.read_text().split(",")]


def get_missing_fractions(missing_fractions) -> List[float]:
    return_value = [float(x) for x in missing_fractions.split(",")]

    for val in return_value:
        if val < 0 or val >= 1:
            raise ValueError(f"valid values for missing_fractions are: 0 <= missing_fraction < 1. Given: {val}")

    return return_value


def get_missing_types(missing_types) -> List[str]:
    return_value = [str(x) for x in missing_types.upper().split(",")]

    for val in return_value:
        if val not in ["MCAR", "MNAR", "MAR"]:
            raise ValueError(f"'{val}' is not a valid missing_type")

    return return_value


def get_id_imputer_class_tuple(task_id: int) -> Tuple[int, OpenMLTask]:

    if task_id in BINARY_TASK_IDS:
        task_class = OpenMLBinaryClassificationTask

    elif task_id in MULTI_TASK_IDS:
        task_class = OpenMLMultiClassClassificationTask

    elif task_id in REGRESSION_TASK_IDS:
        task_class = OpenMLRegressionTask

    else:
        raise ValueError(f"task_id {task_id} isn't supported")

    return task_id, task_class


def get_imputer_class_and_arguments(imputer_name: str) -> BaseImputer:

    if imputer_name not in IMPUTER_CLASS.keys():
        raise ValueError(f"imputer_name '{imputer_name}' not known. Choose one of: {', '.join(IMPUTER_CLASS.keys())}")

    return IMPUTER_CLASS[imputer_name], IMPUTER_ARGUMENTS[imputer_name], IMPUTER_NAME[imputer_name]


def main(
    task_id: int,
    imputer: str,
    experiment_name: str,
    missing_fractions: str = typer.Option(str, help="comma-separated list"),
    missing_types: str = typer.Option(str, help="comma-separated list"),
    base_path: str = "/results"
):

    imputerClass, imputer_arguments, imputer_name = get_imputer_class_and_arguments(imputer.lower())

    experiment_path = Path(base_path) / experiment_name / imputer_name / f"{task_id}"

    if experiment_path.exists():
        raise ValueError(f"Experiment at '{experiment_path}' already exist")

    experiment = Experiment(
        task_id_class_tuples=[get_id_imputer_class_tuple(task_id)],
        missing_fractions=get_missing_fractions(missing_fractions),
        missing_types=get_missing_types(missing_types),
        imputer_class=imputerClass,
        imputer_arguments=imputer_arguments,
        num_repetitions=3,
        base_path=base_path,
        timestamp=experiment_name
    )
    experiment.run()


if __name__ == '__main__':
    typer.run(main)
