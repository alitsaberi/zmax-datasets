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


class ProcessPPG(Transform):
    CHANNEL_NAMES = ["peaks", "ibi", "rate", "quality"]

    def __init__(
        self,
        peak_detection_method: str = PeakDetectionMethod.ELGENDI,
        quality_method: str = QualityMethod.TEMPLATE_MATCH,
        correct_artifacts: bool = False,
        interpolation_method: str = InterpolationMethod.MONOTONE_CUBIC,
    ):
        self.peak_detection_method = peak_detection_method
        self.quality_method = quality_method
        self.correct_artifacts = correct_artifacts
        self.interpolation_method = interpolation_method

    def __call__(self, data: Data) -> Data:
        if data.n_channels != 1:
            raise ValueError(
                "PPG data must have exactly one channel."
                f" Found {data.n_channels} channels."
            )

        ppg_signal = data.array.squeeze()

        peaks, info = nk.ppg_peaks(
            ppg_signal,
            sampling_rate=int(data.sample_rate),
            method=self.peak_detection_method,
            correct_artifacts=self.correct_artifacts,
        )

        periods = nk.signal_period(
            info["PPG_Peaks"],
            sampling_rate=int(data.sample_rate),
            desired_length=len(ppg_signal),
            interpolation_method=self.interpolation_method.value,
        )

        ibi = periods * 1000
        rate = 60 / periods

        # Assess signal quality
        quality = nk.ppg_quality(
            ppg_signal,
            peaks=info["PPG_Peaks"],
            sampling_rate=int(data.sample_rate),
            method=self.quality_method,
        )

        array = np.array([peaks["PPG_Peaks"].values, ibi, rate, quality]).T

        return Data(
            array=array,
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=self.CHANNEL_NAMES,
        )
