import numpy as np

from zmax_datasets.processing.eeg_usability import (
    get_usability_scores,
    load_model,
)
from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class EEGUsability(Transform):
    def __init__(self, model_version: str = "default", return_data: bool = False):
        self._model_version = model_version
        self._model = load_model(model_version)
        self._return_data = return_data

    def __call__(self, data: Data) -> Data:
        """
        Args:
            data (Data): Input data containing EEG left, EEG right, movement.

        Returns:
            Data: EEG usability scores
        """
        if data.n_channels != 3:
            raise ValueError(
                f"Expected 3 channels (EEG left, EEG right, movement), "
                f"got {data.n_channels}"
            )

        usability_scores, samples_to_keep, epoch_length = get_usability_scores(
            data, self._model, *data.channel_names
        )

        if self._return_data:
            data = data[:samples_to_keep]

            # Repeat each label epoch_length times to match data sample rate
            array = np.repeat(usability_scores.array, epoch_length, axis=0)

            # Stack usability scores and data
            array = np.column_stack((array, data.array))

            usability_scores = Data(
                array=array,
                sample_rate=data.sample_rate,
                channel_names=usability_scores.channel_names + data.channel_names,
                timestamps=data.timestamps,
            )

        return usability_scores
