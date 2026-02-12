import numpy as np
from loguru import logger

from zmax_datasets.processing.eeg.artifact_detection.artifact_detection import (
    run_full_zmax_artifact_pipeline_from_data,
)
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




class EEGArtifactDetection(Transform):
    """Artifact detection transform using run_full_zmax_artifact_pipeline_from_data."""

    def __call__(self, data: Data) -> Data:

        n_channels = data.n_channels
        sample_rate = float(data.sample_rate)

        # ---- Single channel case ----
        if n_channels == 1:
            out = run_full_zmax_artifact_pipeline_from_data(data)
            bad = out["artifact_epochs"].astype(float)

            logger.info(
                f"Artifact detection completed: {bad.sum():.0f}/{bad.size} bad epochs"
            )

            return Data(
                array=bad.reshape(-1, 1),
                sample_rate=sample_rate,
                channel_names=[f"bad_epochs_{data.channel_names[0]}"],
            )

        # ---- Multi-channel case ----
        bad_cols = []

        for ch_idx, ch_name in enumerate(data.channel_names):

            # create single-channel Data object
            channel_data = Data(
                array=data.array[:, ch_idx:ch_idx+1],
                sample_rate=sample_rate,
                channel_names=[ch_name],
                timestamps=data.timestamps,
            )

            out= run_full_zmax_artifact_pipeline_from_data(channel_data)
            bad = out["artifact_epochs"].astype(float)

            bad_cols.append(bad)

            logger.info(f"{ch_name}: {bad.sum():.0f}/{bad.size} bad epochs")

        # stack columns
        bad_matrix = np.column_stack(bad_cols)

        return Data(
            array=bad_matrix,
            sample_rate=sample_rate,
            channel_names=[f"bad_epochs_{ch}" for ch in data.channel_names],
        )


