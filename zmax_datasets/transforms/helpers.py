import math
from typing import Literal

import numpy as np
from mne.filter import filter_data
from scipy.signal import resample_poly
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

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


class RepeatResample(Transform):
    """
    Resample data by repeating samples by a given factor.

    This transform increases the sample rate by repeating each sample
    a specified number of times. For example, with factor=2, each sample
    is repeated once, effectively doubling the sample rate.

    Args:
        factor: Integer factor by which to repeat samples (must be >= 1)

    Example:
        If original data has sample rate 100 Hz and factor=2,
        the output will have sample rate 200 Hz with each sample repeated once.
    """

    def __init__(self, factor: int):
        if factor < 1:
            raise ValueError(f"Repeat factor must be >= 1, got {factor}")
        self.factor = factor

    def __call__(self, data: Data) -> Data:
        # Repeat each sample along the time axis (axis=0)
        repeated_array = np.repeat(data.array, self.factor, axis=0)

        return Data(
            array=repeated_array,
            sample_rate=data.sample_rate * self.factor,
            channel_names=data.channel_names,
            timestamp_offset=data.timestamps[0],
        )


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


class Multiply(Transform):
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, data: Data) -> Data:
        return Data(
            array=data.array * self.factor,
            sample_rate=data.sample_rate,
            channel_names=data.channel_names,
            timestamps=data.timestamps,
        )


class TrimToMultiple(Transform):
    """
    Trim data to ensure its duration is divisible by a specified duration.

    This trims from the end of the data to make the total duration
    a multiple of the specified duration.

    Args:
        duration: Duration in seconds that the total duration should be divisible by.

    Example:
        If data is 95 seconds long and duration=10, it will be trimmed to 90 seconds.
    """

    def __init__(self, duration: float):
        self.duration = duration

    def __call__(self, data: Data) -> Data:
        total_samples = data.length
        total_duration = total_samples / data.sample_rate

        target_duration = self.duration * int(total_duration / self.duration)
        target_samples = int(target_duration * data.sample_rate)

        return data[:target_samples]


class Scale(Transform):
    METHODS = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "min_max": MinMaxScaler,
        "max_abs": MaxAbsScaler,
    }

    def __init__(
        self, method: Literal["standard", "robust", "min_max", "max_abs"] = "standard"
    ):
        if method not in self.METHODS:
            raise ValueError(
                f"Invalid scale method: {method}."
                f" Valid methods are: {list(self.METHODS.keys())}"
            )

        self.scaler = self.METHODS[method]()

    def __call__(self, data: Data) -> Data:
        return Data(
            array=self.scaler.fit_transform(data.array),
            sample_rate=data.sample_rate,
            channel_names=data.channel_names,
            timestamps=data.timestamps,
        )


class ClipNoisyValues(Transform):
    """
    Clips all values that are larger than median + iqr_factor * IQR
    or smaller than median - iqr_factor * IQR
    times the IQR of the whole channel.

    This transform identifies and clips extreme values in each channel based on the
    interquartile range (IQR) of that channel. Values exceeding the threshold are
    clipped to the threshold value.

    Args:
        iqr_factor: Factor by which to multiply the IQR to get the threshold.

    Returns:
        Data object with clipped values.
    """

    def __init__(self, iqr_factor: float):
        self.iqr_factor = iqr_factor

    def __call__(self, data: Data) -> Data:
        """
        Apply clipping to noisy values in the data.

        Args:
            data: Input Data object

        Returns:
            Data object with clipped values
        """
        array = data.array.copy()

        median = np.median(array, axis=0)
        percentiles = np.percentile(array, [75, 25], axis=0)
        iqr = np.subtract(percentiles[0], percentiles[1])

        max_threshold = median + iqr * self.iqr_factor
        min_threshold = median - iqr * self.iqr_factor

        array = np.clip(array, min_threshold, max_threshold)

        return Data(
            array=array,
            sample_rate=data.sample_rate,
            channel_names=data.channel_names,
            timestamps=data.timestamps,
        )
