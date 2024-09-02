from collections.abc import Callable

import numpy as np
from scipy.signal import resample_poly


def resample(
    data: np.ndarray,
    sampling_frequency: int,
    old_sampling_frequency: int,
    axis: int = 0,
) -> np.ndarray:
    return resample_poly(data, sampling_frequency, old_sampling_frequency, axis=axis)


def mapper(mapping: dict[int, str]) -> Callable[[np.ndarray, int], np.ndarray]:
    return np.vectorize(mapping.get)
