import math
from typing import Literal

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


class AggregateArtifactMask(Transform):
    """
    Aggregate artifact mask labels over longer windows.

    This is intended for cases where you already have an artifact mask
    at a coarser rate (e.g. one label per 10 seconds, sample_rate=0.1 Hz)
    and want to aggregate multiple consecutive labels into a single label
    (e.g. 30-second labels from 10-second labels).

    The new sample rate will be:
        data.sample_rate / window_size
    so that if you pass window_size=3 for 10s labels (0.1 Hz), you get 30s
    labels at ~0.0333 Hz (one label per 30 seconds).
    """

    def __init__(
        self,
        window_size: int,
        strategy: Literal["first", "last", "majority"] = "majority",
        ignore_last_window: bool = True,
    ):
        """
        Args:
            window_size: Number of consecutive labels to aggregate
                (e.g. 3 to go from 10s labels to 30s labels).
            strategy: How to aggregate labels within each window:
                - "first": use the first label in the window
                - "last": use the last label in the window
                - "majority": use the most frequent value in the window
            ignore_last_window: If True, drop the last incomplete window.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.strategy = strategy
        self.ignore_last_window = ignore_last_window

    def _aggregate_window(self, window: np.ndarray) -> np.ndarray:
        if self.strategy == "first":
            return window[0]
        if self.strategy == "last":
            return window[-1]

        # "majority" vote per channel
        # window shape: (window_size, n_channels)
        aggregated = np.empty(window.shape[1], dtype=window.dtype)
        for ch in range(window.shape[1]):
            values, counts = np.unique(window[:, ch], return_counts=True)
            aggregated[ch] = values[np.argmax(counts)]
        return aggregated

    def __call__(self, data: Data) -> Data:
        """
        Aggregate artifact mask labels along the time axis.

        Expects `data.array` of shape (n_time, n_channels), where channels
        contain artifact mask labels (e.g. 0/1 or categorical).
        """
        n_samples, _ = data.array.shape
        window_size = self.window_size

        if self.ignore_last_window:
            n_windows = n_samples // window_size
        else:
            n_windows = math.ceil(n_samples / window_size)

        if n_windows == 0:
            raise ValueError(
                "Not enough samples to form a single window with "
                f"window_size={window_size} (n_samples={n_samples})"
            )

        aggregated = []
        for i in range(n_windows):
            start = i * window_size
            end = min(start + window_size, n_samples)
            if end - start < window_size and self.ignore_last_window:
                break
            window = data.array[start:end]
            aggregated.append(self._aggregate_window(window))

        aggregated_array = np.stack(aggregated, axis=0)

        return Data(
            array=aggregated_array,
            sample_rate=data.sample_rate / window_size,
            channel_names=data.channel_names,
            timestamp_offset=data.timestamps[0],
        )
