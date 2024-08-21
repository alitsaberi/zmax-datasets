import re
from collections.abc import Callable

import numpy as np
from scipy.signal import resample_poly

from zmax_datasets import settings


def resample(
    data: np.ndarray,
    sampling_frequency: int,
    old_sampling_frequency: int,
    axis: int = 0,
) -> np.ndarray:
    return resample_poly(data, sampling_frequency, old_sampling_frequency, axis=axis)


def mapper(mapping: dict[int, str]) -> Callable[[np.ndarray, int], np.ndarray]:
    return np.vectorize(mapping.get)


def ndarray_to_ids_format(
    stages: np.ndarray, period_length: int = settings.DEFAULTS["period_length"]
):  # TODO: This a uleep util function and should be moved to a usleep-specific place
    num_stages = len(stages)
    initials = np.arange(0, num_stages * period_length, period_length)
    durations = np.full(num_stages, period_length)

    return initials, durations, stages


def squeeze_ids(
    initials: np.ndarray, durations: np.ndarray, annotations: np.ndarray
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:  # TODO: This a uleep util function and should be moved to a usleep-specific place
    changes = np.concatenate(([True], (annotations[1:] != annotations[:-1])))
    squeezed_initials = initials[changes]
    squeezed_annotations = annotations[changes]
    squeezed_durations = np.diff(
        np.concatenate((squeezed_initials, [initials[-1] + durations[-1]]))
    )

    return squeezed_initials, squeezed_durations, squeezed_annotations
