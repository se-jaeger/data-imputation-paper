import numpy as np
import pandas as pd


def _get_search_space_for_grid_search(hyperparameter_grid: dict) -> dict:

    hyperparameters = dict()

    gain_default_hyperparameter_grid = {
        "GAIN": {
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

    # If hyperparameter is given use it, else return default value. All others are ignored.
    for hp_type in gain_default_hyperparameter_grid.keys():
        hyperparameters[hp_type] = {}
        for hp in gain_default_hyperparameter_grid[hp_type].keys():
            hyperparameters[hp_type][hp] = hyperparameter_grid.get(
                hp_type,
                gain_default_hyperparameter_grid[hp_type]
            ).get(
                hp,
                gain_default_hyperparameter_grid[hp_type][hp]
            )

    search_space = dict(
        **hyperparameters["GAIN"],
        **hyperparameters["training"],
        **{f"generator_{key}": value for key, value in hyperparameters["generator"].items()},
        **{f"discriminator_{key}": value for key, value in hyperparameters["discriminator"].items()}
    )

    return search_space


class CategoricalEncoder(object):
    def fit(self, data_frame: pd.DataFrame):

        self._numerical2category = dict()
        self._category2numerical = dict()

        for column in data_frame.columns:
            self._numerical2category[column] = {index: category for index, category in enumerate(data_frame[column].cat.categories)}
            self._category2numerical[column] = {category: index for index, category in enumerate(data_frame[column].cat.categories)}

        return self

    def transform(self, data_frame: pd.DataFrame) -> np.array:

        data_frame = data_frame.copy()

        for column in data_frame.columns:
            data_frame.loc[:, column] = [self._category2numerical[column][value] if not pd.isna(value) else np.nan for value in data_frame[column]]

        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> np.array:

        data_frame = data_frame.copy()

        for column in data_frame.columns:
            data_frame.loc[:, column] = [self._numerical2category[column][value] if not pd.isna(value) else np.nan for value in data_frame[column]]

        return data_frame

    def fit_transform(self, data_frame: pd.DataFrame) -> np.array:
        self.fit(data_frame)
        return self.transform(data_frame)
