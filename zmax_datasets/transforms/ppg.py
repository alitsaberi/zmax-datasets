from enum import Enum

import neurokit2 as nk
import numpy as np

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class PeakDetectionMethod(Enum):
    ELGENDI = "elgendi"
    BISHOP = "bishop"
    CHARLTON = "charlton"


class QualityMethod(Enum):
    TEMPLATE_MATCH = "templatematch"
    DISSIMILARITY = "dissimilarity"


class DetrendMethod(Enum):
    POLYNOMIAL = "polynomial"
    TARVAINEN2002 = "tarvainen2002"
    LOESS = "loess"
    LOCREG = "locreg"


class InterpolationMethod(Enum):
    LINEAR = "linear"
    NEAREST = "nearest"
    ZERO = "zero"
    SLINEAR = "slinear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    PREVIOUS = "previous"
    NEXT = "next"
    MONOTONE_CUBIC = "monotone_cubic"
    AKIMA = "akima"


class PPGProcess(Transform):
    """Process PPG signal to extract peaks, inter-beat intervals, and heart rate.

    Note: Quality assessment has been moved to PPGQuality transform for performance.
    """

    CHANNEL_NAMES = ["peaks", "ibi", "rate"]

    def __init__(
        self,
        peak_detection_method: str = PeakDetectionMethod.ELGENDI,
        correct_artifacts: bool = False,
        interpolation_method: str = InterpolationMethod.MONOTONE_CUBIC,
    ):
        self.peak_detection_method = peak_detection_method
        self.correct_artifacts = correct_artifacts
        self.interpolation_method = interpolation_method

    def __call__(self, data: Data) -> Data:
        if data.n_channels != 1:
            raise ValueError(
                "PPG data must have exactly one channel."
                f" Found {data.n_channels} channels."
            )

        ppg_signal = data.array.squeeze()

        # Peak detection
        peaks, info = nk.ppg_peaks(
            ppg_signal,
            sampling_rate=int(data.sample_rate),
            method=self.peak_detection_method,
            correct_artifacts=self.correct_artifacts,
        )

        # Signal period interpolation
        periods = nk.signal_period(
            info["PPG_Peaks"],
            sampling_rate=int(data.sample_rate),
            desired_length=len(ppg_signal),
            interpolation_method=self.interpolation_method.value,
        )

        ibi = periods * 1000
        rate = 60 / periods

        array = np.array([peaks["PPG_Peaks"].values, ibi, rate]).T

        return Data(
            array=array,
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=self.CHANNEL_NAMES,
        )


class PPGQuality(Transform):
    """Assess PPG signal quality based on peaks.

    This transform takes PPG signal and peaks as input, and outputs quality scores.
    """

    CHANNEL_NAMES = ["quality"]

    def __init__(
        self,
        quality_method: str = QualityMethod.TEMPLATE_MATCH,
    ):
        self.quality_method = quality_method

    def __call__(self, data: Data) -> Data:
        """Assess PPG signal quality.

        Args:
            data (Data): Input data with 2 channels:
                - Channel 0: PPG signal (raw or filtered)
                - Channel 1: Binary peaks signal (1 at peak locations, 0 elsewhere)

        Returns:
            Data: Quality scores (0-1) for each sample.
        """

        if data.n_channels != 2:
            raise ValueError(
                "PPGQuality expects 2 channels (PPG signal, peaks)."
                f" Found {data.n_channels} channels."
            )

        ppg_signal = data.array[:, 0]
        peaks_binary = data.array[:, 1]

        # Convert binary peaks to indices
        peak_indices = np.where(peaks_binary == 1)[0]

        quality = nk.ppg_quality(
            ppg_signal,
            peaks=peak_indices,
            sampling_rate=int(data.sample_rate),
            method=self.quality_method,
        )

        return Data(
            array=quality.reshape(-1, 1),
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=self.CHANNEL_NAMES,
        )


class PPGArtifactLabels(Transform):
    """
    Transform for calculating artifact labels from PPG data
     based on IBI and quality metrics.
    """

    IBI_RANGE = (300, 2000)
    DEFAULT_MIN_VALID_RATIO = 0.5

    def __init__(
        self,
        segment_duration: float,
        quality_threshold: float,
        ibi_range: tuple[float, float] = IBI_RANGE,
        min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
        ignore_last_segment: bool = True,
    ):
        """Initialize PPGArtifactLabels transform.

        Args:
            segment_duration (float): Duration of segments to analyze in seconds.
            quality_threshold (float): Minimum quality score (0-1) for valid segments.
            ibi_range (tuple[float, float]): Valid IBI range in ms
                (e.g., 300-2000ms = 30-200 BPM).
            min_valid_ratio (float): Minimum ratio of valid samples in segment.
            ignore_last_segment (bool):
                Whether to ignore the last segment if it's not full.
        """
        self.segment_duration = segment_duration
        self.quality_threshold = quality_threshold
        self.ibi_range = ibi_range
        self.min_valid_ratio = min_valid_ratio
        self.ignore_last_segment = ignore_last_segment

    def _evaluate_segment(self, ibi: np.ndarray, quality: np.ndarray) -> float:
        """Calculate artifact score for a segment.

        Args:
            ibi (np.ndarray): Inter-beat intervals for the segment in seconds.
            quality (np.ndarray): Quality scores for the segment (0-1).

        Returns:
            float: Artifact score (0-1) where 1 means valid data and 0 means artifact.
        """
        # Check quality threshold
        quality_mask = quality >= self.quality_threshold

        # Check IBI range
        ibi_mask = (ibi >= self.ibi_range[0]) & (ibi <= self.ibi_range[1])

        # Combine masks
        valid_mask = quality_mask & ibi_mask

        # Calculate ratio of valid samples
        valid_ratio = np.mean(valid_mask)

        return valid_ratio

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
            score = self._evaluate_segment(segment_ibi, segment_quality)
            labels.append(score < self.min_valid_ratio)

        # Handle any remaining samples in last segment
        if len(ibi) % samples_per_segment and not self.ignore_last_segment:
            start_idx = n_segments * samples_per_segment
            segment_ibi = ibi[start_idx:]
            segment_quality = quality[start_idx:]
            score = self._evaluate_segment(segment_ibi, segment_quality)
            labels.append(score < self.min_valid_ratio)

        return Data(
            array=np.array(labels).reshape(-1, 1),
            sample_rate=1
            / self.segment_duration,  # Score rate matches segment duration
            channel_names=["artifact_label"],
        )
