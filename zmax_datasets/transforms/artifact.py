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


class IBIArtifactLabels(Transform):
    """
    Transform for calculating artifact labels from IBI data
     based on IBI and quality metrics.
    """

    IBI_RANGE = (300, 2000)

    def __init__(
        self,
        segment_duration: float,
        quality_threshold: float,
        ibi_range: tuple[float, float] = IBI_RANGE,
        ignore_last_segment: bool = True,
    ):
        """Initialize PPGArtifactLabels transform.

        Args:
            segment_duration (float): Duration of segments to analyze in seconds.
            quality_threshold (float): Minimum quality score (0-1) for valid segments.
            ibi_range (tuple[float, float]): Valid IBI range in ms
                (e.g., 300-2000ms = 30-200 BPM).
            ignore_last_segment (bool):
                Whether to ignore the last segment if it's not full.
        """
        self.segment_duration = segment_duration
        self.quality_threshold = quality_threshold
        self.ibi_range = ibi_range
        self.ignore_last_segment = ignore_last_segment

    def _evaluate_segment(self, ibi: np.ndarray, quality: np.ndarray) -> bool:
        """Calculate if a segment is artifactual.

        Args:
            ibi (np.ndarray): Inter-beat intervals for the segment in seconds.
            quality (np.ndarray): Quality scores for the segment (0-1).

        Returns:
            bool: True if the segment is artifactual, False otherwise.
        """
        average_quality = np.nanmean(quality)
        out_of_range_ibi = (ibi < self.ibi_range[0]) | (ibi > self.ibi_range[1])

        return average_quality < self.quality_threshold or np.any(out_of_range_ibi)

    def __call__(self, data: Data) -> Data:
        """Process data and generate artifact labels.

        Args:
            data (Data): Input data containing IBI and quality channels.

        Returns:
            Data: Artifact labels for each segment.
        """
        if data.n_channels != 2:  # IBI, quality
            raise ValueError(
                f"Expected 2 channels (IBI, quality), got {data.n_channels}"
            )

        # Extract channels
        ibi = data.array[:, 0]  # IBI in seconds
        quality = data.array[:, 1]  # Quality scores

        # Calculate samples per segment
        samples_per_segment = int(self.segment_duration * data.sample_rate)
        n_segments = len(ibi) // samples_per_segment

        # Process each segment
        labels = []
        for i in range(n_segments):
            start_idx = i * samples_per_segment
            end_idx = start_idx + samples_per_segment

            # Get segment data
            segment_ibi = ibi[start_idx:end_idx]
            segment_quality = quality[start_idx:end_idx]

            # Calculate segment score
            is_artifactual = self._evaluate_segment(segment_ibi, segment_quality)
            labels.append(int(is_artifactual))

        # Handle any remaining samples in last segment
        if len(ibi) % samples_per_segment and not self.ignore_last_segment:
            start_idx = n_segments * samples_per_segment
            segment_ibi = ibi[start_idx:]
            segment_quality = quality[start_idx:]
            is_artifactual = self._evaluate_segment(segment_ibi, segment_quality)
            labels.append(int(is_artifactual))

        return Data(
            array=np.array(labels).reshape(-1, 1),
            sample_rate=1
            / self.segment_duration,  # Score rate matches segment duration
            channel_names=["artifact_label"],
        )
