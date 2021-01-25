import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam

from ._base import BaseImputer, ImputerError

logger = logging.getLogger()

# TODO: Further Steps:
# 1. compare graph from original paper with this one.
# 2. HPO possibilities


class GAINImputer(BaseImputer):

    def __init__(
        self,
        num_data_columns: int,
        hyperparameters: Dict[str, Union[str, int, float]]  # TODO: check types
    ):

        super().__init__()

        self.num_data_columns = num_data_columns
        self._check_and_set_default_hyperparameters(hyperparameters)

        # GAIN inputs
        X = Input((num_data_columns,))
        M = Input((num_data_columns,))
        H = Input((num_data_columns,))

        # GAIN structure
        self.generator = Sequential(
            [
                Dense(num_data_columns*2, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(num_data_columns, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(num_data_columns, activation=sigmoid, kernel_initializer=GlorotNormal())
            ]
        )

        self.discriminator = Sequential(
            [
                Dense(num_data_columns*2, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(num_data_columns, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(num_data_columns, activation=sigmoid, kernel_initializer=GlorotNormal())
            ]
        )

        gain_input = concatenate([X, M], axis=1)
        generater_output = self.generator(gain_input)
        intermediate = X * M + generater_output * (1 - M)
        intermediate_inputs = concatenate([intermediate, H], axis=1)
        discriminator_output = self.discriminator(intermediate_inputs)

        # GAIN loss
        discriminator_loss = -tf.reduce_mean(M * tf.math.log(discriminator_output + 1e-8) + (1 - M) * tf.math.log(1. - discriminator_output + 1e-8))
        generater_loss_temp = -tf.reduce_mean((1 - M) * tf.math.log(discriminator_output + 1e-8))

        MSE_loss = tf.reduce_mean((M * X - M * generater_output)**2) / tf.reduce_mean(M)
        generater_loss = generater_loss_temp + self.hyperparameters["alpha"] * MSE_loss

        self.gain = Model(inputs=[X, M, H], outputs=[generater_output, generater_loss, discriminator_output, discriminator_loss])

        # preserve original data and add generator output (i.e. imputed values)
        imputer_output = M * X + (1 - M) * generater_output
        self.imputer = Model(inputs=[X, M], outputs=[imputer_output])

    def _prepare_GAIN_input_data(self, data: np.array):
        X_temp = np.nan_to_num(data, nan=0)
        Z_temp = np.random.uniform(0, 0.01, size=[data.shape[0], self.num_data_columns])  # TODO: is this a HP? -> 0.01

        random_hints = np.random.uniform(0., 1., size=[data.shape[0], self.num_data_columns])
        masked_random_hints = 1 * (random_hints < self.hyperparameters["hint_rate"])

        M = 1 - np.isnan(data)
        H = M * masked_random_hints
        X = M * X_temp + (1 - M) * Z_temp

        return X, M, H

    # TODO: check types
    def _check_and_set_default_hyperparameters(self, hyperparameters: Dict[str, Union[str, int, float]]) -> None:

        self.hyperparameters = {
            "alpha": 100,
            "batch_size": 128,
            "epochs": 100,
            "hint_rate": 0.9
        }

    def _encode_data(self, data: pd.DataFrame) -> np.array:

        def _fix_nan(data):
            data = self._categorical_columns_to_string(data)
            data[self._categorical_columns] = SimpleImputer(strategy='constant', fill_value='__NA__').fit_transform(data[self._categorical_columns])

            return data

        missing_mask = data.isna()

        if not self._fitted:

            if self._categorical_columns:

                data = _fix_nan(data)
                self._data_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=len(data))
                data[self._categorical_columns] = self._data_encoder.fit_transform(data[self._categorical_columns])

            self._data_scaler = MinMaxScaler()
            data = self._data_scaler.fit_transform(data)

        else:

            if self._categorical_columns:

                data = _fix_nan(data)
                data[self._categorical_columns] = self._data_encoder.transform(data[self._categorical_columns])

            data = self._data_scaler.transform(data)

        data[missing_mask] = np.nan
        return data

    def _decode_encoded_data(self, encoded_data: np.array, columns: pd.Index, indices: pd.Index) -> pd.DataFrame:

        data = self._data_scaler.inverse_transform(encoded_data)
        data = pd.DataFrame(data, columns=columns, index=indices)

        if self._categorical_columns:

            # round the encoded categories to next int. This is valid because we encode with OrdinalEncoder.
            # clip in range 0..(n-1), where n is the number of categories.
            for index, column in enumerate(self._categorical_columns):
                data[column] = data[column].round(0)
                data[column] = data[column].clip(lower=0, upper=len(self._data_encoder.categories_[index]) - 1)

            data[self._categorical_columns] = self._data_encoder.inverse_transform(data[self._categorical_columns])

        return data

    def fit(self, data: pd.DataFrame, target_columns: List[str], refit: bool = False) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns, refit=refit)

        if data.shape[1] != self.num_data_columns:
            raise ImputerError(f"Given data has {data.shape[1]} columns, expected are {self.num_data_columns}. See constructor.")

        encoded_data = self._encode_data(data.copy())

        generator_optimizer = Adam()
        discriminator_optimizer = Adam()

        generator_var_list = self.generator.trainable_weights
        discriminator_var_list = self.discriminator.trainable_weights

        @tf.function
        def train_step(X, M, H):
            with tf.GradientTape(persistent=True) as tape:
                _, generater_loss, _, discriminator_loss = self.gain([X, M, H])

            generator_gradients = tape.gradient(generater_loss, generator_var_list)
            generator_optimizer.apply_gradients(zip(generator_gradients, generator_var_list))

            discriminator_gradients = tape.gradient(discriminator_loss, discriminator_var_list)
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_var_list))

        train = tf.data.Dataset.from_tensor_slices(encoded_data)
        train_data = train.shuffle(len(train)).batch(self.hyperparameters["batch_size"])

        for _ in range(self.hyperparameters["epochs"]):
            for train_batch in train_data:
                X, M, H = self._prepare_GAIN_input_data(train_batch.numpy())
                train_step(X, M, H)

        self._fitted = True

        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

        imputed_mask = data[self._target_columns].isna().any(axis=1)

        encoded_data = self._encode_data(data.copy())
        X, M, _ = self._prepare_GAIN_input_data(encoded_data)
        imputed = self.imputer([X, M]).numpy()

        # presever everything but the missing values.
        result = data.copy()
        result.loc[imputed_mask, self._target_columns] = self._decode_encoded_data(
            imputed,
            data.columns,
            data.index
        ).loc[imputed_mask, self._target_columns]

        return result, imputed_mask
