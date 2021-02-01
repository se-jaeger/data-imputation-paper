import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam

from ._base import BaseImputer, ImputerError
from .utils import CategoricalEncoder

logger = logging.getLogger()

# TODO: Further Steps:
# 2. HPO possibilities


class GAINImputer(BaseImputer):

    def __init__(
        self,
        num_data_columns: int,
        hyperparameters: Dict[str, Union[str, int, float]],  # TODO: check types
        seed: Optional[int] = None
    ):

        super().__init__(seed=seed)

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
            "batch_size": 256,
            "epochs": 10,
            "hint_rate": 0.9
        }

    def _encode_data(self, data: pd.DataFrame) -> np.array:

        missing_mask = data[self._target_columns].isna()
        column_indices_of_targets = [data.columns.get_loc(column) for column in self._target_columns]

        if not self._fitted:
            if self._categorical_columns:

                self._data_encoder = CategoricalEncoder()
                data[self._categorical_columns] = self._data_encoder.fit_transform(data[self._categorical_columns])

            self._data_scaler = MinMaxScaler()
            data = self._data_scaler.fit_transform(data)

        else:
            if self._categorical_columns:

                data[self._categorical_columns] = self._data_encoder.transform(data[self._categorical_columns])

            data = self._data_scaler.transform(data)

        # NOTE: setting (scattered) values on 2D-arrays is a bit more tricky than on DataFrames
        for missing_mask_index, column_index in enumerate(column_indices_of_targets):
            data[missing_mask.iloc[:, missing_mask_index], column_index] = np.nan

        return data

    def _decode_encoded_data(self, encoded_data: np.array, columns: pd.Index, indices: pd.Index) -> pd.DataFrame:

        data = self._data_scaler.inverse_transform(encoded_data)
        data = pd.DataFrame(data, columns=columns, index=indices)

        if self._categorical_columns:

            # round the encoded categories to next int. This is valid because we encode with OrdinalEncoder.
            # clip in range 0..(n-1), where n is the number of categories.
            for column in self._categorical_columns:
                data[column] = data[column].round(0)
                data[column] = data[column].clip(lower=0, upper=len(self._data_encoder._numerical2category[column].keys()) - 1)

            data[self._categorical_columns] = self._data_encoder.inverse_transform(data[self._categorical_columns])

        return data

    def fit(self, data: pd.DataFrame, target_columns: List[str], refit: bool = False) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns, refit=refit)

        if data.shape[1] != self.num_data_columns:
            raise ImputerError(f"Given data has {data.shape[1]} columns, expected are {self.num_data_columns}. See constructor.")

        encoded_data = self._encode_data(data.copy())

        # TODO: hps
        generator_optimizer = Adam(0.0005)
        discriminator_optimizer = Adam(0.00005)

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

            return generater_loss, discriminator_loss

        train = tf.data.Dataset.from_tensor_slices(encoded_data)
        train_data = train.shuffle(len(train)).batch(self.hyperparameters["batch_size"])

        logger.debug("Start training loop ...")

        for epoch in range(self.hyperparameters["epochs"]):
            total_generator_loss = 0
            total_discriminator_loss = 0

            for train_batch in train_data:
                X, M, H = self._prepare_GAIN_input_data(train_batch.numpy())
                generater_loss, discriminator_loss = train_step(X, M, H)

                total_generator_loss += generater_loss
                total_discriminator_loss += discriminator_loss

            generator_loss_temp = total_generator_loss / self.hyperparameters["batch_size"]
            discriminator_loss_temp = total_discriminator_loss / self.hyperparameters["batch_size"]
            logger.debug(f"Epoch {epoch:>4} losses -- generator: {generator_loss_temp:>6,.4f}; discriminator: {discriminator_loss_temp:>6,.4f}")

        logger.debug("Done training!")

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        imputed_mask = data[self._target_columns].isna()

        encoded_data = self._encode_data(data.copy())
        X, M, _ = self._prepare_GAIN_input_data(encoded_data)
        imputed = self.imputer([X, M]).numpy()

        # presever everything but the missing values in target columns.
        result = data.copy()
        imputed_data_frame = self._decode_encoded_data(imputed, data.columns, data.index)

        for column in imputed_mask.columns:
            result.loc[imputed_mask[column], column] = imputed_data_frame.loc[imputed_mask[column], column]

        return result, imputed_mask
