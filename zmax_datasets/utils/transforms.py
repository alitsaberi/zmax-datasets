import numpy as np
from mne.filter import filter_data
from scipy.signal import resample_poly


def resample(
    data: np.ndarray,
    sampling_frequency: float,
    old_sampling_frequency: float,
    axis: int = 0,
) -> np.ndarray:
    return resample_poly(data, sampling_frequency, old_sampling_frequency, axis=axis)


def l2_normalize(data: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sqrt(np.sum(data**2, axis=axis))


def fir_filter(
    data: np.ndarray,
    sampling_frequency: float,
    low_cutoff: float | None = None,
    high_cutoff: float | None = None,
) -> np.ndarray:
    """
    Apply FIR filter with a Hamming window. The filter length is automatically
    chosen using the filter design function.

    Args:
        data (np.ndarray): The input data to be filtered, shape (..., n_samples).
        sampling_frequency (float): The sampling frequency of the input data in Hz.
        low_cutoff (float, optional): The lower frequency bound of
           the bandpass filter in Hz.
           If None, a highpass filter is applied. Default is None.
        high_cutoff (float, optional): The upper frequency bound of
           the bandpass filter in Hz.
           If None, a lowpass filter is applied. Default is None.

    Returns:
        np.ndarray: The filtered data.

    Notes:
        If both `low_cutoff` and `high_cutoff` are None,
           the original data is returned unchanged.
        If only `low_cutoff` is provided, a highpass filter is applied.
        If only `high_cutoff` is provided, a lowpass filter is applied.
        If `low_cutoff` is greater than `high_cutoff`, a bandstop filter is applied.
        If `low_cutoff` is less than `high_cutoff`, a bandpass filter is applied.
    """
    return filter_data(data, sampling_frequency, low_cutoff, high_cutoff)


def clip_noisy_values(
    data: np.ndarray, min_max_times_global_iqr: int = 20
) -> np.ndarray:
    """
    Clips all values that are larger or smaller than ± min_max_times_global_iqr
    times the IQR of each channel.

    Args:
        data (np.ndarray): A ndarray of shape [..., N] where N is the number of samples.
        sample_frequency (float): The sample rate of data in Hz.
        period_length_sec (float): The length of one epoch/period/segment in seconds.
        min_max_times_global_iqr (int, optional): Extreme value threshold;
                                                  number of times an absolute value
                                                  in a channel must exceed the global
                                                  IQR for that channel to be termed
                                                  an outlier. Defaults to 20.

    Returns:
        np.ndarray: Clipped data, ndarray of shape [C, N]
    """

    # Calculate IQR for each channel
    q75, q25 = np.percentile(data, [75, 25], axis=-1)
    iqr = q75 - q25

    # Reshape bounds to match the input shape, preparing for broadcasting
    lower_bound = np.expand_dims(q25 - min_max_times_global_iqr * iqr, axis=-1)
    upper_bound = np.expand_dims(q75 + min_max_times_global_iqr * iqr, axis=-1)

    # Clip the data
    clipped_data = np.clip(data, lower_bound, upper_bound)

    return clipped_data
