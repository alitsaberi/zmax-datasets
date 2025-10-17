from enum import Enum

import neurokit2 as nk

from zmax_datasets.transforms.base import Transform
from zmax_datasets.transforms.hrv import get_peak_indices
from zmax_datasets.utils.data import Data


class PeakDetectionMethod(Enum):
    ELGENDI = "elgendi"
    BISHOP = "bishop"
    CHARLTON = "charlton"


class QualityMethod(Enum):
    TEMPLATE_MATCH = "templatematch"
    DISSIMILARITY = "dissimilarity"


class PPGPeaks(Transform):
    """
    Process PPG signal to extract peaks.
    """

    CHANNEL_NAMES = ["peaks"]

    def __init__(
        self,
        peak_detection_method: str = PeakDetectionMethod.ELGENDI,
        correct_artifacts: bool = False,
    ):
        self.peak_detection_method = peak_detection_method
        self.correct_artifacts = correct_artifacts

    def __call__(self, data: Data) -> Data:
        if data.n_channels != 1:
            raise ValueError(
                "PPG data must have exactly one channel."
                f" Found {data.n_channels} channels."
            )

        ppg_signal = data.array.squeeze()

        peaks, _ = nk.ppg_peaks(
            ppg_signal,
            sampling_rate=int(data.sample_rate),
            method=self.peak_detection_method,
            correct_artifacts=self.correct_artifacts,
        )

        return Data(
            array=peaks["PPG_Peaks"].values.reshape(-1, 1),
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
        peak_indices = get_peak_indices(peaks_binary)

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
