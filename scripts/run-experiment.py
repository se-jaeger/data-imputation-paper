
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
from data_imputation_paper.imputation.generative import GAINImputer, VAEImputer
from data_imputation_paper.imputation.ml import ForestImputer, KNNImputer
from data_imputation_paper.imputation.simple import ModeImputer

IMPUTER_CLASS = {
    "mode": ModeImputer,
    "knn": KNNImputer,
    "forest": ForestImputer,
    "dl": AutoKerasImputer,
    "gain": GAINImputer,
    "vae": VAEImputer
}

IMPUTER_NAME = {
    "mode": "ModeImputer",
    "knn": "KNNImputer",
    "forest": "ForestImputer",
    "dl": "AutoKerasImputer",
    "gain": "GAINImputer",
    "vae": "VAEImputer"
}

# TODO...
IMPUTER_ARGUMENTS = {
    "mode": {},  # NOTE: there are no arguments here..
    "knn": {
        "hyperparameter_grid_categorical_imputer": {
            "n_neighbors": [1, 3, 5]
        },
        "hyperparameter_grid_numerical_imputer": {
            "n_neighbors": [1, 3, 5]
        }
    },
    "forest": {
        "hyperparameter_grid_categorical_imputer": {
            "n_estimators": [10, 50, 100]
        },
        "hyperparameter_grid_numerical_imputer": {
            "n_estimators": [10, 50, 100]
        }
    },
    "dl": {
        "max_trials": 50,
        "tuner": None,
        "validation_split": 0.2,
        "epochs": 5
    },
    "gain": {
        "hyperparameter_grid": {
            "gain": {
                "alpha": [0.1, 1, 5, 10],
                "hint_rate": [0.7, 0.8, 0.9]
            },
            "training": {
                "epochs": [5, 15]
            },
            "generator": {
                "learning_rate": [0.0005, 0.000005],
            },
            "discriminator": {
                "learning_rate": [0.0005, 0.000005],

            }
        }
    },
    "vae": {
        "hyperparameter_grid": {
            "optimizer": {
                "learning_rate": [0.001]
            },
            "training": {
                "batch_size": [64],
                "epochs": [10],
            },
            # NOTE: Camino's values (http://arxiv.org/abs/1902.10666)
            "neural_architecture": {
                "latent_dim_rel_size": [0.5, 1],
                "n_layers": [0, 1, 2],
                "layer_1_rel_size": [0.5, 1],
                "layer_2_rel_size": [0.5],
            }
        }
    }
}

BINARY_TASK_IDS = [int(x) for x in Path("../data/raw/binary.txt").read_text().split(",")]
MULTI_TASK_IDS = [int(x) for x in Path("../data/raw/multi.txt").read_text().split(",")]
REGRESSION_TASK_IDS = [int(x) for x in Path("../data/raw/regression.txt").read_text().split(",")]


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


def get_strategies(strategies) -> List[str]:
    return_value = [str(x) for x in strategies.lower().split(",")]

    for val in return_value:
        if val not in ["single_single", "multiple_multiple", "single_all", "multiple_all"]:
            raise ValueError(f"'{val}' is not a valid strategies")

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
    strategies: str = typer.Option(str, help="comma-separated list"),
    num_repetitions: int = 10,
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
        strategies=get_strategies(strategies),
        imputer_class=imputerClass,
        imputer_arguments=imputer_arguments,
        num_repetitions=num_repetitions,
        base_path=base_path,
        timestamp=experiment_name
    )
    experiment.run()


if __name__ == '__main__':
    typer.run(main)
