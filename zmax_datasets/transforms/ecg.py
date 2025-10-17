from enum import Enum

import neurokit2 as nk
import numpy as np

from zmax_datasets.transforms.base import Transform
from zmax_datasets.transforms.hrv import get_peak_indices
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


class ECGPeaks(Transform):
    """
    Process ECG signal to extract peaks.
    """

    CHANNEL_NAMES = ["cleaned", "peaks"]

    def __init__(
        self,
        peak_detection_method: str = PeakDetectionMethod.NEUROKIT,
        correct_artifacts: bool = False,
        fix_inversion: bool = True,
    ):
        self.peak_detection_method = peak_detection_method
        self.correct_artifacts = correct_artifacts
        self.fix_inversion = fix_inversion

    def __call__(self, data: Data) -> Data:
        if data.n_channels != 1:
            raise ValueError(
                "ECG data must have exactly one channel."
                f" Found {data.n_channels} channels."
            )

        ecg_signal = data.array.squeeze()

        # Fix inversion
        if self.fix_inversion:
            ecg_signal, _ = nk.ecg_invert(
                ecg_signal, sampling_rate=int(data.sample_rate)
            )

        # Clean ECG signal
        ecg_cleaned = nk.ecg_clean(
            ecg_signal,
            sampling_rate=int(data.sample_rate),
            method=self.peak_detection_method,
        )

        # Peak detection
        peaks, _ = nk.ecg_peaks(
            ecg_cleaned,
            sampling_rate=int(data.sample_rate),
            method=self.peak_detection_method,
            correct_artifacts=self.correct_artifacts,
        )

        array = np.array([ecg_cleaned, peaks["ECG_R_Peaks"].values]).T

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
        r_peaks_indices = get_peak_indices(r_peaks_binary)

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
