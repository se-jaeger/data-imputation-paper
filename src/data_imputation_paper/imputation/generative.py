import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.compat.v1 import logging as tf_logging
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Dense, Input, Layer, concatenate
from tensorflow.keras.optimizers import Adam

from ._base import BaseImputer
from .utils import (
    CategoricalEncoder,
    _get_GAIN_search_space_for_grid_search,
    _get_VAE_search_space_for_grid_search
)

logger = logging.getLogger()
optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.get_logger().setLevel('WARN')
tf_logging.set_verbosity(tf_logging.ERROR)


class GenerativeImputer(BaseImputer):

    def __init__(
        self,
        hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]] = {},
        seed: Optional[int] = None
    ):

        super().__init__(seed=seed)

        self._hyperparameter_grid = hyperparameter_grid
        self._best_hyperparameters = None

    def _encode_data(self, data: pd.DataFrame) -> np.array:
        """
        Encode the input `DataFrame` into an `Array`. Categorical non numerical columns are first encoded, \
            then the whole matrix is scaled to be in range from `0` to `1`.

        Args:
            data (pd.DataFrame): To-be-imputed data

        Returns:
            np.array: To-be-imputed encoded data
        """

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

        return data

    def _decode_encoded_data(self, encoded_data: np.array, columns: pd.Index, indices: pd.Index) -> pd.DataFrame:
        """
        Decodes the imputed data. Takes care of the proper column names and indices of the data.

        Args:
            encoded_data (np.array): Imputed data
            columns (pd.Index): column names of data
            indices (pd.Index): indices of data

        Returns:
            pd.DataFrame: Imputed data as `DataFrame`
        """

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

    def _save_best_imputer(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.value and trial.number == study.best_trial.number) or self._best_hyperparameters is None:
            self.imputer.save(self._model_path, include_optimizer=False)
            self._best_hyperparameters = self.hyperparameters

    def get_best_hyperparameters(self) -> dict:

        super().get_best_hyperparameters()

        return self._best_hyperparameters


class EarlyStoppingExceeded(Exception):
    pass


class EarlyStopping:

    def __init__(self, early_stop: int):
        self.best_loss = None
        self.early_stop_count = 0
        self.early_stop = early_stop

    def update(self, new_loss: float):
        if self.best_loss is None:
            self.best_loss = new_loss
        elif self.best_loss > new_loss:
            self.best_loss = new_loss
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1
            if self.early_stop_count >= self.early_stop:
                raise EarlyStoppingExceeded()


class GAINImputer(GenerativeImputer):

    def __init__(
        self,
        hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]] = {},
        seed: Optional[int] = None
    ):
        """
        Implementation of Generative Adversarial Imputation Nets (GAIN): https://arxiv.org/abs/1806.02920

        Args:
            num_data_columns (int): Number of columns in the to-be-imputed data. Necessary to build the GAIN model properly.
            hyperparameter_grid (Dict[str, Union[int, float, Dict[str, Union[int, float]]]], optional): \
                Provides the hyperparameter grid used for HPO. The dictionary structure is as follows:
                hyperparameter_grid = {
                    "GAIN": {
                        "alpha": [...],
                        "hint_rate": [...],
                        "noise": [...]
                    },
                    "training": {
                        "batch_size": [...],
                        "max_epochs": [...],
                        "early_stop": [...],
                    },
                    "generator": {
                        "learning_rate": [...],
                        "beta_1": [...],
                        "beta_2": [...],
                        "epsilon": [...],
                        "amsgrad": [...]
                    },
                    "discriminator": {
                        "learning_rate": [...],
                        "beta_1": [...],
                        "beta_2": [...],
                        "epsilon": [...],
                        "amsgrad": [...]
                    }
                }
                Defaults to {}.
            seed (Optional[int], optional): To make process as deterministic as possible. Defaults to None.
        """

        super().__init__(
            hyperparameter_grid=hyperparameter_grid,
            seed=seed
        )

        self._model_path = Path("../models/GAIN")

    def _create_GAIN_model(self) -> None:
        """
        Helper method: creates the GAIN model based on the current hyperparameters.
        """

        # GAIN inputs
        X = Input((self._num_data_columns,))
        M = Input((self._num_data_columns,))
        H = Input((self._num_data_columns,))

        # GAIN structure
        self.generator = Sequential(
            [
                Dense(self._num_data_columns*2, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(self._num_data_columns, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(self._num_data_columns, activation=sigmoid, kernel_initializer=GlorotNormal())
            ]
        )

        self.discriminator = Sequential(
            [
                Dense(self._num_data_columns*2, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(self._num_data_columns, activation=relu, kernel_initializer=GlorotNormal()),
                Dense(self._num_data_columns, activation=sigmoid, kernel_initializer=GlorotNormal())
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

        self.gain = Model(inputs=[X, M, H], outputs=[generater_loss, discriminator_loss])

        # preserve original data and add generator output (i.e. imputed values)
        imputer_output = M * X + (1 - M) * generater_output
        self.imputer = Model(inputs=[X, M], outputs=[imputer_output])

    def _train_method(self, trial: optuna.trial.Trial, data: np.array) -> float:
        """
        Optuna's objective function. This is called multiple times by the Optuna framework. Goal is to minimize this method's return values

        Args:
            trial (optuna.trial.Trial): Optuna Trial, a process of evaluating an objective function
            data (np.array): Train data

        Returns:
            float: Generator loss
        """

        # Set hyperparameter once
        self._set_hyperparameters_for_optimization(trial)

        self._create_GAIN_model()

        generator_optimizer = Adam(**self.hyperparameters["generator_Adam"], clipvalue=1)
        discriminator_optimizer = Adam(**self.hyperparameters["discriminator_Adam"], clipvalue=1)

        generator_var_list = self.generator.trainable_weights
        discriminator_var_list = self.discriminator.trainable_weights

        @tf.function
        def train_step(X, M, H):
            with tf.GradientTape(persistent=True) as tape:
                generater_loss, discriminator_loss = self.gain([X, M, H])

            generator_gradients = tape.gradient(generater_loss, generator_var_list)
            generator_optimizer.apply_gradients(zip(generator_gradients, generator_var_list))

            discriminator_gradients = tape.gradient(discriminator_loss, discriminator_var_list)
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_var_list))

            return generater_loss, discriminator_loss

        n_splits = 3
        k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=self._seed)
        cv_generator_loss = 0

        for train_index, test_index in k_fold.split(data):

            train = tf.data.Dataset.from_tensor_slices(data[train_index])
            train_data = train.batch(self.hyperparameters["batch_size"])

            early_stopping = EarlyStopping(self.hyperparameters["early_stop"])

            for _ in range(self.hyperparameters["max_epochs"]):
                for train_batch in train_data:
                    X, M, H = self._prepare_GAIN_input_data(train_batch.numpy())
                    train_step(X, M, H)

                X_train, M_train, H_train = self._prepare_GAIN_input_data(data[train_index])
                epoch_loss, _ = self.gain([X_train, M_train, H_train])
                try:
                    early_stopping.update(epoch_loss)
                except EarlyStoppingExceeded:
                    break

            X, M, H = self._prepare_GAIN_input_data(data[test_index])
            generater_loss, _ = self.gain([X, M, H])
            cv_generator_loss += generater_loss

        # Optuna minimizes/maximizes this return value
        return cv_generator_loss / n_splits

    def _prepare_GAIN_input_data(self, data: np.array) -> Tuple[np.array, np.array, np.array]:
        """
        Prepare data to use it as GAIN input.

        Args:
            data (np.array): Encoded and (0, 1) scaled data

        Returns:
            Tuple[np.array, np.array, np.array]: Three matrices all of the same shapes used as GAIN input: \
                `X` (data matrix), `M` (mask matrix), `H` (hint matrix)
        """

        X_temp = np.nan_to_num(data, nan=0)
        Z_temp = np.random.uniform(0, self.hyperparameters["noise"], size=[data.shape[0], self._num_data_columns])

        random_hints = np.random.uniform(0., 1., size=[data.shape[0], self._num_data_columns])
        masked_random_hints = 1 * (random_hints < self.hyperparameters["hint_rate"])

        M = 1 - np.isnan(data)
        H = M * masked_random_hints
        X = M * X_temp + (1 - M) * Z_temp

        return X, M, H

    def _set_hyperparameters_for_optimization(self, trial: optuna.trial.Trial) -> None:
        """
        Helper method: samples hyperparameters for a HPO process

        Args:
            trial (optuna.trial.Trial): Optuna Trial, a process of evaluating an objective function
        """

        self.hyperparameters = {
            # GAIN
            "alpha": trial.suggest_discrete_uniform("alpha", 0, 9999999999, 1),
            "hint_rate": trial.suggest_discrete_uniform("hint_rate", 0, 1, 1),
            "noise": trial.suggest_discrete_uniform("noise", 0, 1, 1),

            # training
            "batch_size": trial.suggest_discrete_uniform("batch_size", 0, 1024, 1),
            "max_epochs": trial.suggest_discrete_uniform("max_epochs", 0, 10000, 1),
            "early_stop": trial.suggest_discrete_uniform("early_stop", 0, 1000, 1),

            # optimizers
            "generator_Adam": {
                "learning_rate": trial.suggest_discrete_uniform("generator_learning_rate", 0, 1, 1),
                "beta_1": trial.suggest_discrete_uniform("generator_beta_1", 0, 1, 1),
                "beta_2": trial.suggest_discrete_uniform("generator_beta_2", 0, 1, 1),
                "epsilon": trial.suggest_discrete_uniform("generator_epsilon", 0, 1, 1),
                "amsgrad": trial.suggest_categorical("generator_amsgrad", [True, False])
            },
            "discriminator_Adam": {
                "learning_rate": trial.suggest_discrete_uniform("discriminator_learning_rate", 0, 1, 1),
                "beta_1": trial.suggest_discrete_uniform("discriminator_beta_1", 0, 1, 1),
                "beta_2": trial.suggest_discrete_uniform("discriminator_beta_2", 0, 1, 1),
                "epsilon": trial.suggest_discrete_uniform("discriminator_epsilon", 0, 1, 1),
                "amsgrad": trial.suggest_categorical("discriminator_amsgrad", [True, False])
            }
        }

    def fit(self, data: pd.DataFrame, target_columns: List[str]) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns)

        self._num_data_columns = data.shape[1]

        encoded_data = self._encode_data(data.copy())

        search_space = _get_GAIN_search_space_for_grid_search(self._hyperparameter_grid)
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="minimize")
        study.optimize(
            lambda trial: self._train_method(trial, encoded_data),
            callbacks=[self._save_best_imputer]
        )  # NOTE: n_jobs=-1 causes troubles because TensorFlow shares the graph across processes

        if self._model_path.exists():
            self.imputer = tf.keras.models.load_model(self._model_path, compile=False)
            shutil.rmtree(self._model_path, ignore_errors=True)

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        super().transform(data=data)

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


class VAESampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEImputer(GenerativeImputer):

    def __init__(
        self,
        hyperparameter_grid: Dict[str, Dict[str, List[Union[int, float, bool]]]] = {},
        seed: Optional[int] = None
    ):
        """
        Implementation of Variational Autoencoder.

        Args:
            num_data_columns (int): Number of columns in the to-be-imputed data. Necessary to build the VAE model properly.
            hyperparameter_grid (Dict[str, Union[int, float, Dict[str, Union[int, float]]]], optional): \
                Provides the hyperparameter grid used for HPO. The dictionary structure is as follows:
                hyperparameter_grid = {
                    "optimizer": {
                        "learning_rate": [...],
                        "beta_1": [...],
                        "beta_2": [...],
                        "epsilon": [...],
                        "amsgrad": [...]
                    },
                    "training": {
                        "batch_size": [...],
                        "max_epochs": [...],
                        "early_stop": [...],
                    },
                    "neural_architecture": {
                        "latent_dim_rel_size": [...],
                        "n_layers": [...],
                        "layer_1_rel_size": [...],
                        "layer_2_rel_size": [...],
            }
                }
                Defaults to {}.
            seed (Optional[int], optional): To make process as deterministic as possible. Defaults to None.
        """

        super().__init__(
            hyperparameter_grid=hyperparameter_grid,
            seed=seed
        )

        self._model_path = Path("../models/VAE")

    def _create_VAE_model(self) -> None:
        """
        Helper method: creates the VAE model based on the current hyperparameters.
        """

        latent_dim = int(
            round(
                self.hyperparameters["latent_dim_rel_size"] * self._num_data_columns, 0
            )
        )
        n_layers = self.hyperparameters["n_layers"]

        # Build the encoder
        encoder_inputs = Input((self._num_data_columns,))
        if n_layers == 0:
            x = encoder_inputs
        else:
            for i in range(n_layers):
                layer = i + 1
                units = int(round(self.hyperparameters[f"layer_{layer}_rel_size"] * self._num_data_columns, 0))
                if i == 0:
                    x = Dense(units, activation=relu)(encoder_inputs)
                else:
                    x = Dense(units, activation=relu)(x)
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = VAESampling()([z_mean, z_log_var])

        # Build the decoder
        if n_layers == 0:
            x = z
        else:
            for i in range(n_layers):
                layer = n_layers - i
                units = int(round(self.hyperparameters[f"layer_{layer}_rel_size"] * self._num_data_columns, 0))
                if i == 0:
                    x = Dense(units, activation=relu)(z)
                else:
                    x = Dense(units, activation=relu)(x)
        decoder_outputs = Dense(self._num_data_columns, activation=sigmoid)(x)

        # VAE loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(encoder_inputs, decoder_outputs)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        # self.imputer = Model(inputs=encoder_inputs, outputs=[decoder_outputs, total_loss])
        self.imputer = Model(inputs=encoder_inputs, outputs=decoder_outputs)  # -> .transform()
        self.trainable_model = Model(inputs=encoder_inputs, outputs=total_loss)  # -> .fit()

    def _train_method(self, trial: optuna.trial.Trial, data: np.array) -> float:
        """
        Optuna's objective function. This is called multiple times by the Optuna framework.
        Goal is to minimize this method's return values.

        Args:
            trial (optuna.trial.Trial): Optuna Trial, a process of evaluating an objective function
            data (np.array): Train data

        Returns:
            float: Generator loss
        """

        # Set hyperparameter once
        self._set_hyperparameters_for_optimization(trial)

        self._create_VAE_model()

        optimizer = Adam(**self.hyperparameters["optimizer_Adam"])

        @tf.function
        def train_step(data):
            with tf.GradientTape() as tape:
                total_loss = self.trainable_model(data)
            grads = tape.gradient(total_loss, self.trainable_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.trainable_model.trainable_weights))
            return total_loss

        n_splits = 3
        k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=self._seed)
        cv_loss = 0

        for train_index, test_index in k_fold.split(data):

            train = tf.data.Dataset.from_tensor_slices(data[train_index])
            train_data = train.batch(self.hyperparameters["batch_size"])

            early_stopping = EarlyStopping(self.hyperparameters["early_stop"])

            for i in range(self.hyperparameters["max_epochs"]):
                for train_batch in train_data:
                    X = self._prepare_VAE_input_data(train_batch.numpy())
                    train_step(X)

                X_train = self._prepare_VAE_input_data(data[train_index])
                epoch_loss = self.trainable_model(X_train)
                try:
                    early_stopping.update(epoch_loss)
                except EarlyStoppingExceeded:
                    break

            X = self._prepare_VAE_input_data(data[test_index])
            loss = self.trainable_model(X)
            cv_loss += loss

        # Optuna minimizes/maximizes this return value
        return cv_loss / n_splits

    def _prepare_VAE_input_data(self, data: np.array) -> Tuple[np.array, np.array, np.array]:
        """
        Prepare data to use it as VAE input.

        Args:
            data (np.array): Encoded and (0, 1) scaled data

        Returns:
            Tuple[np.array, np.array, np.array]: Three matrices all of the same shapes used as GAIN input: \
                `X` (data matrix), `M` (mask matrix), `H` (hint matrix)
        """
        X_temp = np.nan_to_num(data, nan=0)
        Z_temp = np.random.uniform(0, 0.01, size=[data.shape[0], self._num_data_columns])

        M = 1 - np.isnan(data)
        X = M * X_temp + (1 - M) * Z_temp

        return X

    def _set_hyperparameters_for_optimization(self, trial: optuna.trial.Trial) -> None:
        """
        Helper method: samples hyperparameters for a HPO process

        Args:
            trial (optuna.trial.Trial): Optuna Trial, a process of evaluating an objective function
        """

        hyperparameters = {
            # training
            "batch_size": trial.suggest_discrete_uniform("batch_size", 0, 1024, 1),
            "max_epochs": trial.suggest_discrete_uniform("max_epochs", 0, 10000, 1),
            "early_stop": trial.suggest_discrete_uniform("early_stop", 0, 1000, 1),

            # optimizer
            "optimizer_Adam": {
                "learning_rate": trial.suggest_discrete_uniform("optimizer_learning_rate", 0, 1, 1),
                "beta_1": trial.suggest_discrete_uniform("optimizer_beta_1", 0, 1, 1),
                "beta_2": trial.suggest_discrete_uniform("optimizer_beta_2", 0, 1, 1),
                "epsilon": trial.suggest_discrete_uniform("optimizer_epsilon", 0, 1, 1),
                "amsgrad": trial.suggest_categorical("optimizer_amsgrad", [True, False])
            },
        }

        # neural architecture
        n_layers = trial.suggest_int("n_layers", 0, 2)
        neural_architecture = {
            "latent_dim_rel_size": trial.suggest_float("latent_dim_rel_size", 0, 1),
            "n_layers": n_layers,
        }
        for i in range(n_layers):
            layer = i + 1
            neural_architecture[f"layer_{layer}_rel_size"] = trial.suggest_float(f"layer_{layer}_rel_size", 0, 1)
        hyperparameters = {**hyperparameters, **neural_architecture}

        self.hyperparameters = hyperparameters

    def fit(self, data: pd.DataFrame, target_columns: List[str]) -> BaseImputer:

        super().fit(data=data, target_columns=target_columns)

        self._num_data_columns = data.shape[1]

        encoded_data = self._encode_data(data.copy())

        search_space = _get_VAE_search_space_for_grid_search(self._hyperparameter_grid)
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction="minimize")
        study.optimize(
            lambda trial: self._train_method(trial, encoded_data),
            callbacks=[self._save_best_imputer]
        )  # NOTE: n_jobs=-1 causes troubles because TensorFlow shares the graph across processes

        if self._model_path.exists():
            self.imputer = tf.keras.models.load_model(self._model_path, compile=False)
            shutil.rmtree(self._model_path, ignore_errors=True)

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        super().transform(data=data)

        imputed_mask = data[self._target_columns].isna()

        encoded_data = self._encode_data(data.copy())

        X = self._prepare_VAE_input_data(encoded_data)
        imputed = self.imputer([X]).numpy()

        # presever everything but the missing values in target columns.
        result = data.copy()
        imputed_data_frame = self._decode_encoded_data(imputed, data.columns, data.index)
        for column in imputed_mask.columns:
            result.loc[imputed_mask[column], column] = imputed_data_frame.loc[imputed_mask[column], column]

        return result, imputed_mask
