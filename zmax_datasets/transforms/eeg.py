import asyncio

import numpy as np
from loguru import logger

from zmax_datasets.processing.eeg.quality import (
    score_amplitude,
    score_bandpower,
    score_burstiness,
    score_flatness,
    score_kurtosis,
)
from zmax_datasets.processing.eeg.usability import (
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


class EEGQuality(Transform):
    """
    Transform for computing signal quality scores for EEG channels.

    This transform segments the input data and computes multiple quality metrics
    including flatness, amplitude, bandpower, and burstiness scores.
    """

    SEGMENT_DURATION = 10.0
    FLATNESS_WINDOW_DURATION = 1.0
    STD_THRESHOLD = 0.5
    MAX_ABSOLUTE_VALUE = 500.0
    MIN_FREQUENCY = 0.3
    MAX_FREQUENCY = 20.0
    P2P_THRESHOLD = 30.0
    BURSTINESS_WINDOW_DURATION = 0.2

    METRIC_NAMES = [
        "flatness",
        "amplitude",
        "bandpower_score",
        "bandpower_value",
        "burstiness",
        "kurtosis",
    ]

    def __init__(
        self,
        segment_duration: float = SEGMENT_DURATION,
        flatness_window_duration: float = FLATNESS_WINDOW_DURATION,
        std_threshold: float = STD_THRESHOLD,
        max_absolute_value: float = MAX_ABSOLUTE_VALUE,
        min_frequency: float = MIN_FREQUENCY,
        max_frequency: float = MAX_FREQUENCY,
        p2p_threshold: float = P2P_THRESHOLD,
        burstiness_window_duration: float = BURSTINESS_WINDOW_DURATION,
    ):
        """
        Initialize the EEGQuality transform.

        Args:
            segment_duration: Duration of each segment in seconds
            flatness_window_duration:
                Duration of windows for flatness detection in seconds
            std_threshold: Minimum std deviation for "good" windows
            max_absolute_value: Maximum allowed amplitude in microvolts
            min_frequency: Lower bound for EEG frequency band in Hz
            max_frequency: Upper bound for EEG frequency band in Hz
            p2p_threshold:
                Minimum p2p amplitude to consider a burst in microvolts
            burstiness_window_duration:
                Duration of windows for burstiness detection in seconds
        """
        self.segment_duration = segment_duration
        self.flatness_window_duration = flatness_window_duration
        self.std_threshold = std_threshold
        self.max_absolute_value = max_absolute_value
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.p2p_threshold = p2p_threshold
        self.burstiness_window_duration = burstiness_window_duration

    def __call__(self, data: Data) -> Data:
        """
        Compute signal quality scores for EEG channels.

        Each segment is scored independently and returned with corresponding timestamps.

        Args:
            data: EEG data with shape (n_samples, n_channels)

        Returns:
            Data object with shape (n_segments, n_channels * 6) where each
            original channel has 6 metrics: flatness, amplitude, bandpower_score,
            bandpower_value, burstiness, and kurtosis
        """
        return asyncio.run(self._process(data))

    async def _process(self, data: Data) -> Data:
        """
        Internal async method to process the data.

        Args:
            data: EEG data with shape (n_samples, n_channels)

        Returns:
            Data object with quality scores
        """
        n_samples, n_channels = data.shape
        segment_length = int(self.segment_duration * data.sample_rate)
        n_segments = n_samples // segment_length

        if n_segments == 0:
            logger.warning(
                f"Data too short for {self.segment_duration}s segments. "
                "Returning empty results."
            )
            return self._create_empty_results(n_channels, data.channel_names)

        # Reshape data into segments
        segments = data.array[: n_segments * segment_length].reshape(
            n_segments, segment_length, n_channels
        )

        # Initialize arrays for results
        flatness_scores = np.zeros((n_segments, n_channels))
        amplitude_scores = np.zeros((n_segments, n_channels))
        bandpower_scores = np.zeros((n_segments, n_channels))
        bandpower_values = np.zeros((n_segments, n_channels))
        burstiness_scores = np.zeros((n_segments, n_channels))
        kurtosis_scores = np.zeros((n_segments, n_channels))

        # Process all segments in parallel
        segment_results = await asyncio.gather(
            *[self._score_segment(segment, data.sample_rate) for segment in segments]
        )

        # Unpack results
        for i, (
            flatness_score,
            amplitude_score,
            bandpower_score,
            bandpower_value,
            burstiness_score,
            kurtosis_score,
        ) in enumerate(segment_results):
            flatness_scores[i] = flatness_score
            amplitude_scores[i] = amplitude_score
            bandpower_scores[i] = bandpower_score
            bandpower_values[i] = bandpower_value
            burstiness_scores[i] = burstiness_score
            kurtosis_scores[i] = kurtosis_score

        # Calculate segment timestamps (middle of each segment)
        segment_timestamps = np.array(
            [
                data.timestamps[i * segment_length + segment_length // 2]
                for i in range(n_segments)
            ]
        )

        # Concatenate all metrics into a single array
        # Shape: (n_segments, n_channels * 6)
        combined_array = np.concatenate(
            [
                flatness_scores,
                amplitude_scores,
                bandpower_scores,
                bandpower_values,
                burstiness_scores,
                kurtosis_scores,
            ],
            axis=1,
        )

        combined_channel_names = [
            f"{channel_name}_{metric}"
            for channel_name in data.channel_names
            for metric in self.METRIC_NAMES
        ]

        # Create single Data object with all metrics
        segment_sample_rate = 1.0 / self.segment_duration

        return Data(
            array=combined_array,
            sample_rate=segment_sample_rate,
            channel_names=combined_channel_names,
            timestamps=segment_timestamps,
        )

    async def _score_segment(
        self, segment: np.ndarray, sample_rate: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Score a single segment for all quality metrics.

        Args:
            segment: EEG segment array with shape (segment_length, n_channels)
            sample_rate: Sampling rate in Hz

        Returns:
            Tuple of scores:
                (flatness, amplitude, bandpower_score,
                bandpower_value, burstiness, kurtosis)
        """

        def _compute_scores():
            # Create temporary Data object for flatness scoring
            segment_data = Data(
                array=segment,
                sample_rate=sample_rate,
            )

            flatness_score = score_flatness(
                segment_data,
                self.flatness_window_duration,
                self.std_threshold,
            )

            amplitude_score = score_amplitude(
                segment_data,
                self.max_absolute_value,
            )

            bandpower_score, bandpower_value = score_bandpower(
                segment_data,
                self.min_frequency,
                self.max_frequency,
                reverse=False,
            )

            burstiness_score = score_burstiness(
                segment_data,
                self.burstiness_window_duration,
                self.p2p_threshold,
            )

            kurtosis_score = score_kurtosis(segment_data)

            return (
                flatness_score,
                amplitude_score,
                bandpower_score,
                bandpower_value,
                burstiness_score,
                kurtosis_score,
            )

        return await asyncio.to_thread(_compute_scores)

    def _create_empty_results(self, n_channels: int, channel_names: list[str]) -> Data:
        """
        Create empty Data object when input is too short.

        Args:
            n_channels: Number of channels in the input data
            channel_names: Names of the input channels

        Returns:
            Empty Data object with proper channel names for all metrics
        """
        segment_sample_rate = 1.0 / self.segment_duration

        combined_channel_names = [
            f"{channel_name}_{metric}"
            for channel_name in channel_names
            for metric in self.METRIC_NAMES
        ]

        # Empty array with shape (0, n_channels * 6)
        empty_array = np.empty((0, n_channels * len(self.METRIC_NAMES)))
        empty_timestamps = np.array([])

        return Data(
            array=empty_array,
            sample_rate=segment_sample_rate,
            channel_names=combined_channel_names,
            timestamps=empty_timestamps,
        )
