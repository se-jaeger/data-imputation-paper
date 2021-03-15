import random
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf


def set_seed(seed: int) -> None:
    if seed:
        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


def _get_GAIN_search_space_for_grid_search(
    hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]]
) -> Dict[str, List[Union[int, float, bool]]]:

    gain_default_hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]] = {
        "gain": {
            "alpha": [100],
            "hint_rate": [0.9],
            "noise": [0.01]
        },
        "training": {
            "batch_size": [64],
            "max_epochs": [50],
            "early_stop": [3]
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

    hyperparameters = _merge_given_HPs_with_defaults(hyperparameter_grid, gain_default_hyperparameter_grid)

    search_space = dict(
        **hyperparameters["gain"],
        **hyperparameters["training"],
        **{f"generator_{key}": value for key, value in hyperparameters["generator"].items()},
        **{f"discriminator_{key}": value for key, value in hyperparameters["discriminator"].items()}
    )

    return search_space


def _get_VAE_search_space_for_grid_search(
    hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]]
) -> Dict[str, List[Union[int, float, bool]]]:

    vae_default_hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]] = {
        "training": {
            "batch_size": [64],
            "max_epochs": [50],
            "early_stop": [3]
        },
        "optimizer": {
            "learning_rate": [0.001],
            "beta_1": [0.9],
            "beta_2": [0.999],
            "epsilon": [1e-7],
            "amsgrad": [False]
        },
        "neural_architecture": {
            "latent_dim_rel_size": [0.5, 0.2],
            "n_layers": [0, 1, 2],
            "layer_1_rel_size": [0.75, 0.5],
            "layer_2_rel_size": [0.25],
        }
    }

    hyperparameters = _merge_given_HPs_with_defaults(hyperparameter_grid, vae_default_hyperparameter_grid)

    search_space = dict(
        **hyperparameters["training"],
        **{f"optimizer_{key}": value for key, value in hyperparameters["optimizer"].items()},
        **hyperparameters["neural_architecture"],
    )

    return search_space


def _merge_given_HPs_with_defaults(
    hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]],
    default_hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]]
) -> Dict[str, Dict[str, List[Union[int, float, bool]]]]:
    """
    If hyperparameter is given use it, else return default value. All others are ignored.
    """
    hyperparameters: Dict[str, Dict[str, List[Union[int, float, bool]]]] = dict()
    for hp_type in default_hyperparameter_grid.keys():
        hp_type = hp_type.lower()
        hyperparameters[hp_type] = {}
        for hp in default_hyperparameter_grid[hp_type].keys():
            hp = hp.lower()
            hyperparameters[hp_type][hp] = hyperparameter_grid.get(
                hp_type,
                default_hyperparameter_grid[hp_type]
            ).get(
                hp,
                default_hyperparameter_grid[hp_type][hp]
            )
    return hyperparameters


class CategoricalEncoder(object):
    """
    Encoder only works on categorical columns. \
        It encodes the categorical values into `int` values fro `0` `n - 1`, where `n` is the number of categories.
    """

    def fit(self, data_frame: pd.DataFrame):
        """
        Creates attributes that map categorical values to integers and vice versa, separately for each column.

        Args:
            data_frame (pd.DataFrame): Data to fit mappings.

        Returns:
            CategoricalEncoder: Instance of itself.
        """

        self._numerical2category = dict()
        self._category2numerical = dict()

        for column in data_frame.columns:
            self._numerical2category[column] = {index: category for index, category in enumerate(data_frame[column].cat.categories)}
            self._category2numerical[column] = {category: index for index, category in enumerate(data_frame[column].cat.categories)}

        return self

    def transform(self, data_frame: pd.DataFrame) -> np.array:
        """
        Maps each column to their int representations.

        Args:
            data_frame (pd.DataFrame): To-be-encoded data

        Returns:
            np.array: Encoded data as matrix
        """

        data_frame = data_frame.copy()

        for column in data_frame.columns:
            data_frame.loc[:, column] = [self._category2numerical[column][value] if not pd.isna(value) else np.nan for value in data_frame[column]]

        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> np.array:
        """
        Maps each int value to its categorical representations.

        Args:
            data_frame (pd.DataFrame): To-be-decoded data

        Returns:
            np.array: Decoded data as matrix
        """

        data_frame = data_frame.copy()

        for column in data_frame.columns:
            data_frame.loc[:, column] = [self._numerical2category[column][value] if not pd.isna(value) else np.nan for value in data_frame[column]]

        return data_frame

    def fit_transform(self, data_frame: pd.DataFrame) -> np.array:
        """
        Combines fitting of representations and returns (a copy) of their encoded values.

        Args:
            data_frame (pd.DataFrame): To-be-fitted and transformed data

        Returns:
            np.array: Encoded data as matrix
        """

        return self.fit(data_frame).transform(data_frame)
