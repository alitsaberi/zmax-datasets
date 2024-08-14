import re

import numpy as np
from scipy.signal import resample_poly


def extract_id_by_regex(name: str, regex: re.Pattern) -> int | None:
    match = regex.search(name)
    return int(match.group("id")) if match else None


def resample(
    data: np.ndarray,
    sampling_frequency: int,
    old_sampling_frequency: int,
    axis: int = 0,
) -> np.ndarray:
    return resample_poly(data, sampling_frequency, old_sampling_frequency, axis=axis)


def rescale_and_clip_data(data: np.ndarray) -> np.ndarray:
    median = np.median(data, axis=-1, keepdims=True)
    q1 = np.percentile(data, 25, axis=-1, keepdims=True)
    q3 = np.percentile(data, 75, axis=-1, keepdims=True)
    iqr = q3 - q1

    # Clip values with absolute deviation from median more than 20 * IQR
    threshold = 20 * iqr
    data_clipped = np.clip(data, median - threshold, median + threshold)

    # To avoid division by zero in case IQR is zero, add a small epsilon
    iqr[iqr == 0] = 1e-6

    return (data_clipped - median) / iqr
