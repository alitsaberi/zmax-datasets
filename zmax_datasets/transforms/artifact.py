import numpy as np
from loguru import logger

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class MergeArtifactLabels(Transform):
    """Transform for merging artifact labels with annotations."""

    def __init__(
        self,
        artifact_label: str = "ARTIFACT",
    ):
        """
        Args:
            artifact_label (str): Label value to use for artifact segments.
        """
        self.artifact_label = artifact_label

    def __call__(self, data: Data) -> Data:
        """Merge artifact labels with annotations.

        Args:
            data (Data): Input data containing annotations and artifact masks.

        Returns:
            Data: Updated annotations with artifact labels.
        """
        if data.n_channels != 2:  # annotations, artifact_mask
            raise ValueError(
                "Expected 2 channels (annotations, artifact_mask),"
                f" got {data.n_channels}"
            )

        annotations = data.array[:, 0].copy()  # Make a copy to avoid modifying input
        artifact_mask = data.array[:, 1].squeeze()

        logger.info(f"Number of artifact segments: {np.sum(artifact_mask)}")

        annotations[artifact_mask.astype(bool)] = self.artifact_label

        return Data(
            array=annotations.reshape(-1, 1),
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=["annotations"],
        )
