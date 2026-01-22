from loguru import logger
import numpy as np

from zmax_datasets.processing.eeg_artifact_detect import (
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
    """Rule-based EEG artifact labeling per 30s epoch.

    Output is one sample per epoch with:
        - bad_L: 1 if left channel is bad, else 0
        - bad_R: 1 if right channel is bad, else 0
        - artifact_any: 1 if either channel is bad, else 0
    """

    def __init__(
        self,
        left_label: str = "EEG_L",
        right_label: str = "EEG_R",
        epoch_length: float = 30.0,
        plot: bool = False,
    ):
        self.left_label = left_label
        self.right_label = right_label
        self.epoch_length = epoch_length
        self.plot = plot
        

    def __call__(self, data: Data) -> Data:
        out = run_full_zmax_artifact_pipeline_from_data(
            data=data,
            left_label=self.left_label,
            right_label=self.right_label,
            sf=float(data.sample_rate),
            plot_eeg=self.plot)

        bad_L = out["bad_L"]
        bad_R = out["bad_R"]

        artifact_any = ((bad_L == 1) | (bad_R == 1)).astype(int)

        n_epochs = bad_L.shape[0]
        n_artifacts = int(artifact_any.sum())
        logger.info(f"Artifact epochs (any channel bad): {n_artifacts}/{n_epochs}")

        # 3) Pack into Data: (n_epochs, 3)
        arr = np.column_stack([bad_L, bad_R, artifact_any])

        return Data(
            array=arr,
            sample_rate=1.0 / self.epoch_length,
            channel_names=["bad_L", "bad_R", "artifact_any"],
        )

