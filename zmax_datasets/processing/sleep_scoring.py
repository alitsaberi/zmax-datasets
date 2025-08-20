import copy
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from psg_utils.preprocessing import apply_scaling, quality_control_funcs
from utime import Defaults
from utime.hyperparameters import YAMLHParams

from zmax_datasets.transforms.resample import Resample
from zmax_datasets.utils.data import Data, get_all_periods

if TYPE_CHECKING:
    from tensorflow.keras.models import Model

SLEEP_STAGE_CHANNEL_NAME = "sleep_stage"


def read_hyperparameters_file(path: Path) -> YAMLHParams:
    """
    Read hyperparameters from a YAML file.
    """
    return YAMLHParams(path, no_version_control=True)


class UTimeModel:
    def __init__(
        self,
        model_dir: Path,
        weight_file_name: str | None = None,
        n_periods: int | None = None,
        n_samples_per_prediction: int | None = None,
    ):
        """
        Initializes a UTimeModel object.
        Args:
            model_dir (Path): The utime project directory.
            weight_file_name (str): The name of the weight file in models directory.
            n_periods (int): The number of periods to use for prediction.
                             If None, the number used for training is used.
            n_samples_per_prediction (int): Number of samples that should make up
                                            each sleep stage scoring.
                                            Defaults to n_samples_per_periods,
                                            giving 1 segmentation per period
                                            of signal. Set this to 1 to score
                                            every data point in the signal.
        """

        self._model_dir = model_dir
        self._weight_file_name = weight_file_name
        self._n_periods = n_periods
        self._n_samples_per_prediction = n_samples_per_prediction
        self._model = None
        self._hyperparameters = None
        self._dataset = None
        self.load()
        self._resample = Resample(self.input_sample_rate)

    def __repr__(self) -> str:
        return (
            f"UTimeModel(model_dir={self._model_dir},"
            f" input_shape={self.input_shape},"
            f" n_samples_per_prediction={self.n_samples_per_prediction})"
        )

    @property
    def name(self) -> str:
        return self._model_dir.name

    @property
    def hyperparameters(self) -> YAMLHParams:
        return self._hyperparameters

    @property
    def model(self) -> "Model":
        return self._model

    @property
    def n_samples_per_prediction(self) -> int:
        return self._model.data_per_prediction

    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Returns the input shape of the model.
        n_periods, n_samples_per_periods, n_channels
        """
        return self._model.layers[0].input_shape[0][1:]

    @property
    def input_sample_rate(self) -> int:
        return self.hyperparameters["set_sample_rate"]

    @property
    def n_periods(self) -> int:
        return self.input_shape[0]

    @n_periods.setter
    def n_periods(self, value: int) -> None:
        self._n_periods = value
        self.load()

    @property
    def n_samples_per_period(self) -> int:
        return self.input_shape[1]

    @property
    def period_duration(self) -> float:
        """
        Returns the duration of a period in seconds.
        """
        return self.n_samples_per_period / self.input_sample_rate

    @property
    def stage_mapping(self) -> dict[str, int]:
        return self.hyperparameters["sleep_stage_annotations"]

    @cached_property
    def sleep_stage_annotations(self) -> list[str]:
        inverse_stage_mapping = {v: k for k, v in self.stage_mapping.items()}
        return [
            inverse_stage_mapping[i]
            for i in range(self.hyperparameters["build"]["n_classes"])
        ]

    def load(self) -> None:
        self._hyperparameters = self._load_hyperparameters()
        self._model = self._load_model()
        logger.debug(f"Loading model {self!r}")

    def _load_hyperparameters(self) -> YAMLHParams:
        """
        Loads the hyperparameters from the model directory.
        """
        hyperparameters_path = Defaults.get_hparams_path(self._model_dir)
        hyperparameters = read_hyperparameters_file(hyperparameters_path)

        self._update_build_hyperparameters(hyperparameters)

        if datasets := hyperparameters.get("datasets"):
            self._dataset = list(datasets.keys())[0]
            self._update_dataset_hyperparameters(hyperparameters)

        return hyperparameters

    def _update_build_hyperparameters(self, hyperparameters: YAMLHParams) -> None:
        if hyperparameters["build"].get("batch_shape") is None:
            logger.debug(
                "Build hyperparameters batch_shape not set, checking for preprocessed"
                " hyperparameters"
            )
            preprocessed_hyperparameters_path = Path(
                Defaults.get_pre_processed_hparams_path(self._model_dir)
            )

            if not preprocessed_hyperparameters_path.exists():
                raise ValueError(
                    f"Failed to set batch_shape for model {self.name}: Could not find"
                    " preprocessed hyperparameters at"
                    f" {preprocessed_hyperparameters_path} to set batch_shape"
                )

            preprocessed_hyperparameters = read_hyperparameters_file(
                preprocessed_hyperparameters_path
            )

            if preprocessed_hyperparameters["build"].get("batch_shape") is None:
                raise ValueError(
                    f"Failed to set batch_shape for model {self.name}:"
                    " no batch_shape found in hyperparameters"
                )

            hyperparameters["build"]["batch_shape"] = preprocessed_hyperparameters[
                "build"
            ]["batch_shape"]

        if self._n_periods is not None:
            hyperparameters["build"]["batch_shape"][1] = self._n_periods

        if self._n_samples_per_prediction is not None:
            hyperparameters["build"]["data_per_prediction"] = (
                self._n_samples_per_prediction
            )

    def _update_dataset_hyperparameters(self, hyperparameters: YAMLHParams) -> None:
        """
        Updates the dataset hyperparameters with the dataset configuration.
        """
        dataset_configuration_path = Path(hyperparameters["datasets"][self._dataset])

        if not dataset_configuration_path.is_absolute():
            dataset_configuration_path = (
                Defaults.get_hparams_dir(self._model_dir) / dataset_configuration_path
            )

        logger.debug(f"Loading dataset configuration from {dataset_configuration_path}")

        dataset_configuration = read_hyperparameters_file(dataset_configuration_path)
        hyperparameters.update(dataset_configuration)

    def _load_model(self) -> "Model":
        from utime.bin.evaluate import get_and_load_model

        return get_and_load_model(
            self._model_dir,
            self.hyperparameters,
            self._weight_file_name,
            clear_previous=True,
        )

    def prepare_data(self, data: Data) -> np.ndarray:
        """
        Prepares the data for prediction.
        """

        if self._hyperparameters is None:
            raise ValueError(
                "Model not loaded. Please call load() before preparing data."
            )

        data = copy.deepcopy(data)

        if (n_samples_dropped := data.length % self.n_samples_per_period) != 0:
            logger.warning(
                f"Dropping {n_samples_dropped} samples to match model input shape"
                f" requirement of {self.n_samples_per_period} samples per period."
            )
            data = data[:-n_samples_dropped]

        if data.sample_rate != self.input_sample_rate:
            data = self._resample(data, self.input_sample_rate)

        if quality_control := self.hyperparameters.get("quality_control_func"):
            # Run over epochs and assess if epoch-specific changes should be
            # made to limit the influence of very high noise level ecochs etc.
            _apply_quality_control(
                data,
                **quality_control,
                period_length_sec=self.n_samples_per_period / data.sample_rate,
            )

        if scaler := self.hyperparameters.get("scaler"):
            _apply_scaling(data, scaler)

        periods = get_all_periods(data, self.n_samples_per_period)

        batch_size = periods.shape[0] // self.n_periods

        if (n_periods_dropped := periods.shape[0] % self.n_periods) != 0:
            logger.warning(
                f"Dropping {n_periods_dropped} periods to match model input shape"
                f" requirement of {self.input_shape[0]} periods per batch."
            )
            periods = periods[:-n_periods_dropped]

        return periods.reshape(
            batch_size, self.n_periods, self.n_samples_per_period, -1
        ).astype(np.float32)

    def predict(self, data: np.ndarray, channel_groups: list[list[int]]) -> np.ndarray:
        """
        Predicts the sleep stage for the given data.

        Args:
            data (np.ndarray): The data to predict on.
            channel_groups (list[list[int]]): A list of channel groups.
                                              Each group contains the indices
                                              of the channels to be used for
                                              prediction. The final prediction
                                              is the sum of the predictions
                                              for each group.
        """

        if self.model is None:
            raise ValueError("Model not loaded. Please call load() before predicting.")

        self._assert_channel_groups(channel_groups)

        predictions_list = []

        for i, channel_group in enumerate(channel_groups):
            logger.debug(
                f"Predicting for channel group {i} with channels {channel_group}"
            )
            current_predictions = self.model.predict_on_batch(data[..., channel_group])
            predictions_list.append(current_predictions)

        predictions = np.mean(predictions_list, axis=0)
        predictions = predictions.reshape(-1, predictions.shape[-1])
        return predictions

    def _assert_channel_groups(self, channel_groups: list[list[int]]) -> None:
        """
        Asserts that the channel groups are valid.
        """
        if not channel_groups:
            raise ValueError("No channel groups provided")

        for i, group in enumerate(channel_groups):
            if len(group) == 0:
                raise ValueError(f"Channel group {i} is empty.")

            if len(group) != self.input_shape[-1]:
                raise ValueError(
                    f"Channel group {group} includes more channels than the expected"
                    f" input shape {self.input_shape}"
                )


def _apply_quality_control(data: Data, quality_control_func: str, **kwargs) -> Data:
    quality_control_function = getattr(quality_control_funcs, quality_control_func)
    data.array, indices = quality_control_function(
        psg=data.array,
        sample_rate=data.sample_rate,
        **kwargs,
    )

    for i, affected_periods in enumerate(indices):
        logger.info(
            f"Quality control affected {len(affected_periods)}"
            f" periods in channel {i}"
        )

    return data


def _apply_scaling(data: Data, scaler: str) -> Data:
    data.array, _ = apply_scaling(data.array, scaler)
    return data


def score(
    data: Data,
    model: UTimeModel,
    channel_groups: list[list[int | str]] | None = None,
    arg_max: bool = True,
) -> Data:
    """
    Score sleep stages from ZMax data using a UTime model.

    Args:
        data: Input Data object containing ZMax data
        model: Trained UTimeModel for sleep staging
        channel_groups: List of channel groups to use for prediction.
                        If None, uses individual channels for each group.
        arg_max: Whether to return class predictions (True)
                 or class probabilities (False)

    Returns:
        Data: Object containing sleep stage predictions or probabilities
              if arg_max is False, it includes sleep stage annotations.
    """
    channel_groups = channel_groups or [list(range(data.n_channels))]

    if isinstance(channel_groups[0][0], str):
        channel_groups = [data.get_channel_indices(group) for group in channel_groups]

    output_sample_rate = model.input_sample_rate / model.n_samples_per_prediction
    timestamp_offset = np.mean(
        data.timestamps[
            : int(
                (model.n_samples_per_prediction / model.input_sample_rate)
                * data.sample_rate
            )
        ]
    ).astype(np.int64)

    data = model.prepare_data(data)
    logger.info(
        f"Predicting sleep stages for data with shape {data.shape}"
        f" and channel groups {channel_groups}"
    )
    predictions = model.predict(data, channel_groups)
    channel_names = model.sleep_stage_annotations

    if arg_max:
        predictions = np.reshape(np.argmax(predictions, axis=-1), (-1, 1))
        channel_names = [SLEEP_STAGE_CHANNEL_NAME]
        logger.debug(f"Applied argmax, prediction shape: {predictions.shape}")

    return Data(
        predictions,
        sample_rate=output_sample_rate,
        channel_names=channel_names,
        timestamp_offset=timestamp_offset,
    )
