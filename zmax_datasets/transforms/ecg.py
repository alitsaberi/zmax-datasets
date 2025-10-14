from enum import Enum

import neurokit2 as nk
import numpy as np

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class PeakDetectionMethod(Enum):
    NEUROKIT = "neurokit"
    PAN_TOMPKINS = "pantompkins"
    HAMILTON = "hamilton"
    ELGENDI = "elgendi"
    ENGZEE = "engzee"
    VISIBILITY_GRAPH = "vg"


class QualityMethod(Enum):
    TEMPLATE_MATCH = "templatematch"
    AVERAGE_QRS = "averageQRS"
    ZHAO = "zhao2018"


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


class ECGProcess(Transform):
    """
    Process ECG signal to extract peaks, inter-beat intervals, and heart rate.
    """

    CHANNEL_NAMES = ["cleaned", "peaks", "ibi", "rate"]

    def __init__(
        self,
        peak_detection_method: str = PeakDetectionMethod.NEUROKIT,
        correct_artifacts: bool = False,
        interpolation_method: str = InterpolationMethod.MONOTONE_CUBIC,
        invert_signal: bool = False,
    ):
        self.peak_detection_method = peak_detection_method
        self.correct_artifacts = correct_artifacts
        self.interpolation_method = interpolation_method
        self.invert_signal = invert_signal

    def __call__(self, data: Data) -> Data:
        if data.n_channels != 1:
            raise ValueError(
                "ECG data must have exactly one channel."
                f" Found {data.n_channels} channels."
            )

        ecg_signal = data.array.squeeze()

        if self.invert_signal:
            ecg_signal = -ecg_signal

        # Clean ECG signal
        ecg_cleaned = nk.ecg_clean(
            ecg_signal,
            sampling_rate=int(data.sample_rate),
            method=self.peak_detection_method,
        )

        # Peak detection
        peaks, info = nk.ecg_peaks(
            ecg_signal,
            sampling_rate=int(data.sample_rate),
            method=self.peak_detection_method,
            correct_artifacts=self.correct_artifacts,
        )

        # Signal period interpolation
        periods = nk.signal_period(
            info["ECG_R_Peaks"],
            sampling_rate=int(data.sample_rate),
            desired_length=len(ecg_signal),
            interpolation_method=self.interpolation_method.value,
        )

        ibi = periods * 1000
        rate = 60 / periods

        array = np.array([ecg_cleaned, peaks["ECG_R_Peaks"].values, ibi, rate]).T

        return Data(
            array=array,
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=self.CHANNEL_NAMES,
        )


class ECGQuality(Transform):
    """Assess ECG signal quality based on peaks.

    This transform takes ECG signal and R-peaks as input, and outputs quality scores.
    """

    CHANNEL_NAMES = ["quality"]

    def __init__(
        self,
        quality_method: str = QualityMethod.AVERAGE_QRS,
    ):
        self.quality_method = quality_method

    def __call__(self, data: Data) -> Data:
        """Assess PPG signal quality.

        Args:
            data (Data): Input data with 2 channels:
                - Channel 0: ECG signal (filtered)
                - Channel 1: Binary R-peaks signal (1 at R-peak locations, 0 elsewhere)

        Returns:
            Data:
                - Quality scores (0-1) for each sample
                    (`averageQRS` and `templatematch`)
                - String classification (Unacceptable,
                    Barely acceptable or Excellent) (`zhao2018`)
        """

        if data.n_channels != 2:
            raise ValueError(
                "ECGQuality expects 2 channels (ECG signal, peaks)."
                f" Found {data.n_channels} channels."
            )

        ecg_signal = data.array[:, 0]
        r_peaks_binary = data.array[:, 1]

        # Convert binary peaks to indices
        r_peaks_indices = np.where(r_peaks_binary == 1)[0]

        quality = nk.ecg_quality(
            ecg_signal,
            rpeaks=r_peaks_indices,
            sampling_rate=int(data.sample_rate),
            method=self.quality_method,
        )

        return Data(
            array=quality.reshape(-1, 1),
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=self.CHANNEL_NAMES,
        )
