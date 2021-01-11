from typing import Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ._base import SklearnBaseImputer


class ForestImputer(SklearnBaseImputer):

    def __init__(
        self,
        grid_categorical_imputer_arguments: Dict[str, object] = {},
        grid_numerical_imputer_arguments: Dict[str, object] = {},
        categorical_precision_threshold: float = 0.85
    ):
        super().__init__(
            (RandomForestClassifier(n_jobs=-1), grid_categorical_imputer_arguments),
            (RandomForestRegressor(n_jobs=-1), grid_numerical_imputer_arguments),
            categorical_precision_threshold=categorical_precision_threshold
        )


class KNNImputer(SklearnBaseImputer):

    def __init__(
        self,
        grid_categorical_imputer_arguments: Dict[str, object] = {},
        grid_numerical_imputer_arguments: Dict[str, object] = {},
        categorical_precision_threshold: float = 0.85
    ):
        super().__init__(
            (KNeighborsClassifier(n_jobs=-1), grid_categorical_imputer_arguments),
            (KNeighborsRegressor(n_jobs=-1), grid_numerical_imputer_arguments),
            categorical_precision_threshold=categorical_precision_threshold
        )
