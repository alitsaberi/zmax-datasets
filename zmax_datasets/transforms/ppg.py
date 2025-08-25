from enum import Enum

import neurokit2 as nk
import numpy as np
from scipy.interpolate import interp1d

from zmax_datasets.processing.ibi import extract_ibi_from_peaks
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


class ProcessPPG(Transform):
    CHANNEL_NAMES = ["peaks", "rate", "quality"]

    def __init__(
        self,
        peak_detection_method: str = PeakDetectionMethod.ELGENDI,
        quality_method: str = QualityMethod.TEMPLATE_MATCH,
        correct_artifacts: bool = False,
    ):
        self.peak_detection_method = peak_detection_method
        self.quality_method = quality_method
        self.correct_artifacts = correct_artifacts

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
            show=False,
        )

        # Rate computation
        rate = nk.signal_rate(
            info["PPG_Peaks"],
            sampling_rate=int(data.sample_rate),
            desired_length=len(ppg_signal),
        )

        # Assess signal quality
        quality = nk.ppg_quality(
            ppg_signal,
            peaks=info["PPG_Peaks"],
            sampling_rate=int(data.sample_rate),
            method=self.quality_method,
        )

        array = np.array([peaks["PPG_Peaks"].values, rate, quality]).T

        return Data(
            array=array,
            sample_rate=data.sample_rate,
            timestamps=data.timestamps,
            channel_names=self.CHANNEL_NAMES,
        )


class ExtractIBI(Transform):
    CHANNEL_NAMES = ["ibi"]

    def __init__(
        self,
        interpolation_rate: float,
        detrend: DetrendMethod | None = None,
        match_length: bool = True,
    ):
        self.interpolation_rate = interpolation_rate
        self.detrend = detrend
        self.match_length = match_length

    def __call__(self, peaks: Data, **kwargs) -> Data:
        if peaks.n_channels != 1:
            raise ValueError(
                "PPG peaks data must have exactly one channel."
                f" Found {peaks.n_channels} channels."
            )

        peaks_signal = peaks.array.squeeze()
        ibi_values, ibi_times = extract_ibi_from_peaks(peaks_signal, peaks.sample_rate)

        # Process IBI values
        ibi_signal, ibi_times, _ = nk.intervals_process(
            ibi_values,
            intervals_time=ibi_times,
            interpolate=True,
            interpolation_rate=self.interpolation_rate,
            detrend=self.detrend,
            **kwargs,
        )

        # Ensure the length matches peaks by interpolating to exact timestamps
        if self.match_length and len(ibi_signal) != len(peaks_signal):
            f = interp1d(
                ibi_times, ibi_signal, bounds_error=False, fill_value="extrapolate"
            )

            # Generate timestamps matching peaks length
            target_times = np.arange(len(peaks_signal)) / peaks.sample_rate
            ibi_signal = f(target_times)

        return Data(
            array=ibi_signal.reshape(-1, 1),
            sample_rate=peaks.sample_rate,
            timestamps=peaks.timestamps,
            channel_names=["ibi"],
        )
