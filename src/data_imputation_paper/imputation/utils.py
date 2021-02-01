import numpy as np
import pandas as pd


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
