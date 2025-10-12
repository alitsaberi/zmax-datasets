from loguru import logger

from zmax_datasets.processing.eeg_usability import (
    get_usability_scores,
    load_model,
)
from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class EEGUsability(Transform):
    def __init__(self, model_version: str = "default"):
        self._model_version = model_version
        self._model = load_model(model_version)

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

        usability_scores, _, _ = get_usability_scores(
            data, self._model, *data.channel_names
        )

        logger.info(
            "EEG left artifact count:"
            f" {usability_scores[:, 0].array.sum()}/{usability_scores.length}%"
        )
        logger.info(
            "EEG right artifact count:"
            f" {usability_scores[:, 1].array.sum()}/{usability_scores.length}%"
        )

        return usability_scores
