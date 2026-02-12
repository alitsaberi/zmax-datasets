# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 12:06:43 2026

@author: selaca
"""
import numpy as np
from loguru import logger
from typing import Any, Literal
from zmax_datasets.utils.data import Data
from zmax_datasets.utils.exceptions import (
    SampleRateMismatchError,
    InvalidEpochDurationError,
    EpochLengthTooSmallError,
    IncompleteEpochsError,
    NoSamplesError,
)

def epochify(
    data: Data,
    *,
    epoch_duration: float,
    expected_sample_rate: float | None = None,
    drop_remainder: bool = True,
) -> tuple[np.ndarray, int, int, Data]:
    """Convert a continuous Data object into fixed-length epochs.

    Args:
        data: Input continuous data with shape (n_samples, n_channels).
        epoch_duration: Epoch duration in seconds.
        expected_sample_rate: If provided, validate that
            ``data.sample_rate`` matches this value.
        drop_remainder: If True, drop samples at the end that do not make a
            complete epoch. If False, raise an error when the data length is
            not an exact multiple of the epoch length.

    Returns:
        A tuple containing:
            - array: NumPy array of shape (n_epochs, n_samples_per_epoch, n_channels)
            - n_epochs: Number of epochs.
            - epoch_length: Number of samples per epoch.
            - trimmed_data: Input data trimmed so that it contains only
              full epochs.

    Raises:
        ValueError: If ``epoch_duration`` is not positive, if
            ``expected_sample_rate`` does not match ``data.sample_rate``,
            or if ``drop_remainder`` is False and the data length is not
            a multiple of the epoch length.
        NoSamplesError: If no complete epochs can be formed from the data.
    """
    
    if expected_sample_rate is not None and data.sample_rate != expected_sample_rate:
        raise SampleRateMismatchError(expected_sample_rate, data.sample_rate)

    if epoch_duration <= 0:
        raise InvalidEpochDurationError(epoch_duration)

    epoch_length = int(epoch_duration * data.sample_rate)
    if epoch_length <= 0:
        raise EpochLengthTooSmallError(epoch_duration, data.sample_rate)

    n_epochs = data.length // epoch_length
    if n_epochs == 0:
        raise NoSamplesError(data.length, epoch_length)

    samples_to_keep = n_epochs * epoch_length

    logger.debug(
    "samples={}, epoch_length={}, n_epochs={}, samples_to_keep={}",
    data.length,
    epoch_length,
    n_epochs,
    samples_to_keep,
    )

    
    # TODO: support padding incomplete epochs (e.g., zero- or NaN-padding)
    if samples_to_keep < data.length:
        if not drop_remainder:
            raise IncompleteEpochsError(data.length, epoch_length)
        logger.debug(
            "Dropping {} samples at the end",
            data.length - samples_to_keep,
        )

        data = data[:samples_to_keep]

    array = data.array.reshape(
        n_epochs, epoch_length, data.n_channels
    )  # (N, T, C)

    return array, n_epochs, epoch_length, data



def detect_voltage_unit(
    values: np.ndarray,
    *,
    microvolt_threshold: float = 1e-3,
) -> Literal["volts", "microvolts"]:
    """Heuristically detect whether a signal is in volts or microvolts.

    Detection is based on the absolute magnitude of the signal.
    This is a heuristic and should not be considered authoritative.

    Args:
        values: Signal values.
        microvolt_threshold: Threshold above which values are assumed
            to be in microvolts. Defaults to 1e-3.

    Returns:
        "volts" if the signal appears to be in volts,
        "microvolts" if the signal appears to be in microvolts.
    """
    max_abs = np.nanmax(np.abs(values))

    if max_abs > microvolt_threshold:
        return "microvolts"

    return "volts"



def microvolts_to_volts(values: np.ndarray) -> np.ndarray:
    """Convert microvolts (µV) to volts (V).

    Args:
        values: Array of values in microvolts.

    Returns:
        Array of values in volts.
    """
    return values * 1e-6


def volts_to_microvolts(values: np.ndarray) -> np.ndarray:
    """Convert volts (V) to microvolts (µV).

    Args:
        values: Array of values in volts.

    Returns:
        Array of values in microvolts.
    """
    return values * 1e6

def ensure_volts(values: np.ndarray) -> np.ndarray:
    """Ensure signal is in volts, converting from microvolts if needed."""
    unit = detect_voltage_unit(values)
    return microvolts_to_volts(values) if unit == "microvolts" else values

def ensure_microvolts(values: np.ndarray) -> np.ndarray:
    """Ensure signal is in microvolts, converting from volts if needed."""
    unit = detect_voltage_unit(values)
    return volts_to_microvolts(values) if unit == "volts" else values