import math

import numpy as np
from mne.filter import filter_data
from scipy.signal import resample_poly

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data


class Resample(Transform):
    def __init__(self, new_sample_rate: float):
        self.new_sample_rate = new_sample_rate

    def __call__(
        self,
        data: Data,
        **kwargs,
    ) -> Data:
        # Calculate resampling factors
        gcd = math.gcd(int(self.new_sample_rate * 1000), int(data.sample_rate * 1000))
        up = int(self.new_sample_rate * 1000) // gcd
        down = int(data.sample_rate * 1000) // gcd

        # Resample using integer factors
        resampled = Data(
            resample_poly(
                data.array.astype(np.float64),
                up,
                down,
                axis=0,
                **kwargs,
            ),
            sample_rate=self.new_sample_rate,
            channel_names=data.channel_names,
            timestamp_offset=data.timestamps[0],
        )

        return resampled


class FIRFilter(Transform):
    """
    Apply FIR filter with a Hamming window. The filter length is automatically
    chosen using the filter design function.

    If both `low_cutoff` and `high_cutoff` are None,
    the original data is returned unchanged.
    If only `low_cutoff` is provided, a highpass filter is applied.
    If only `high_cutoff` is provided, a lowpass filter is applied.
    If `low_cutoff` is greater than `high_cutoff`, a bandstop filter is applied.
    If `low_cutoff` is less than `high_cutoff`, a bandpass filter is applied.
    """

    def __init__(
        self,
        low_cutoff: float | None = None,
        high_cutoff: float | None = None,
    ):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def __call__(
        self,
        data: Data,
        **kwargs,
    ) -> Data:
        return Data(
            filter_data(
                data.array.T,
                data.sample_rate,
                self.low_cutoff,
                self.high_cutoff,
                **kwargs,
            ).T,
            sample_rate=data.sample_rate,
            channel_names=data.channel_names,
            timestamps=data.timestamps,
        )


class Scale(Transform):
    def __init__(self, scale_factor: float):
        self.scale_factor = scale_factor

    def __call__(self, data: Data) -> Data:
        return Data(
            array=data.array * self.scale_factor,
            sample_rate=data.sample_rate,
            channel_names=data.channel_names,
            timestamps=data.timestamps,
        )
