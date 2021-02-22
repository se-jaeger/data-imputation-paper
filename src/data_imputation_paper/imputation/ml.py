from typing import Dict, Optional
import logging

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ._base import SklearnBaseImputer


logger = logging.getLogger()


class ForestImputer(SklearnBaseImputer):

    def __init__(
        self,
        hyperparameter_grid_categorical_imputer: Dict[str, object] = {},
        hyperparameter_grid_numerical_imputer: Dict[str, object] = {},
        seed: Optional[int] = None
    ):
        """
        Imputer uses `RandomForestClassifier` for categorical columns and `RandomForestRegressor` for numerical columns.

        Args:
            hyperparameter_grid_categorical_imputer (Dict[str, object], optional): \
                Hyperparameter grid for HPO for `RandomForestClassifier`. Defaults to {}.
            hyperparameter_grid_numerical_imputer (Dict[str, object], optional): \
                Hyperparameter grid for HPO for `RandomForestRegressor`. Defaults to {}.
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        super().__init__(
            (RandomForestClassifier(n_jobs=-1), hyperparameter_grid_categorical_imputer),
            (RandomForestRegressor(n_jobs=-1), hyperparameter_grid_numerical_imputer),
            seed=seed
        )


class KNNImputer(SklearnBaseImputer):

    def __init__(
        self,
        hyperparameter_grid_categorical_imputer: Dict[str, object] = {},
        hyperparameter_grid_numerical_imputer: Dict[str, object] = {},
        seed: Optional[int] = None
    ):
        """
        Imputer uses `KNeighborsClassifier` for categorical columns and `KNeighborsRegressor` for numerical columns.

        Args:
            hyperparameter_grid_categorical_imputer (Dict[str, object], optional): \
                Hyperparameter grid for HPO for `KNeighborsClassifier`. Defaults to {}.
            hyperparameter_grid_numerical_imputer (Dict[str, object], optional): \
                Hyperparameter grid for HPO for `KNeighborsRegressor`. Defaults to {}.
            seed (Optional[int], optional): Seed to make behavior deterministic. Defaults to None.
        """

        super().__init__(
            (KNeighborsClassifier(n_jobs=-1), hyperparameter_grid_categorical_imputer),
            (KNeighborsRegressor(n_jobs=-1), hyperparameter_grid_numerical_imputer),
            seed=seed
        )
