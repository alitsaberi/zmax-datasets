import numpy as np
from loguru import logger
from scipy.signal import welch
from scipy.stats import kurtosis

from zmax_datasets.utils.data import Data


def score_flatness(
    data: Data,
    window_duration: float,
    std_threshold: float,
) -> np.ndarray:
    """
    Scores channels based on how often the signal's standard deviation exceeds
    a threshold in small windows — detects flat signals.

    Args:
        data: Data object.
        window_duration: Duration of window in seconds
        std_threshold: Minimum standard deviation to consider window "good"

    Returns:
        score: Fraction of good windows per channel, shape (n_channels,), range [0, 1]
    """
    n_samples = data.length
    n_channels = data.n_channels
    window_samples = int(window_duration * data.sample_rate)
    n_windows = n_samples // window_samples

    if n_windows == 0:
        logger.warning(
            f"No windows found in the data. Data length ({n_samples})"
            f" is too short to create a window of size"
            f" {window_duration} seconds."
        )
        return np.zeros(n_channels)

    reshaped = data[: n_windows * window_samples].array.reshape(
        n_windows, window_samples, n_channels
    )
    stds = np.std(reshaped, axis=1)  # shape (n_windows, n_channels)
    good_windows = stds >= std_threshold
    return np.mean(good_windows, axis=0)


def score_amplitude(data: Data, max_absolute_value: float) -> np.ndarray:
    """
    Scores channels based on how often their absolute amplitude is below a threshold.

    Args:
        data: Data object.
        max_absolute_value: Maximum allowed absolute amplitude

    Returns:
        score: Fraction of time points within the amplitude limit,
          shape (n_channels,), range [0, 1]
    """
    return np.mean(np.abs(data.array) <= max_absolute_value, axis=0)


def score_bandpower(
    data: Data,
    min_frequency: float,
    max_frequency: float,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scores channels based on the ratio of band power to total power.

    Args:
        data: Data object.
        min_frequency: Band start (Hz)
        max_frequency: Band end (Hz)
        reverse: If True, returns 1 - band ratio (useful for detecting noise bands)

    Returns:
        score: Ratio score per channel, shape (n_channels,), range [0, 1]
    """
    n_samples, n_channels = data.shape
    scores = np.zeros(n_channels)
    band_powers = np.zeros(n_channels)
    for channel in range(n_channels):
        freqs, psd = welch(
            data[:, channel].array.squeeze(),
            fs=data.sample_rate,
            nperseg=min(2 * data.sample_rate, n_samples),
        )
        total_power = np.trapz(psd, freqs)
        band_mask = (freqs >= min_frequency) & (freqs <= max_frequency)
        band_power = np.trapz(psd[band_mask], freqs[band_mask])
        band_powers[channel] = band_power

        ratio = band_power / (total_power + np.finfo(float).eps)
        score = np.clip(ratio, 0.0, 1.0)
        scores[channel] = 1.0 - score if reverse else score

    return scores, band_powers


def score_kurtosis(data: Data) -> np.ndarray:
    """
    Scores channels based on their kurtosis.

    Args:
        data: Data object.

    Returns:
        score: Kurtosis score per channel, shape (n_channels,), range [0, 1]
    """
    return kurtosis(data.array, axis=0, fisher=False)


def score_burstiness(
    data: Data,
    window_duration: float,
    p2p_threshold: float,
) -> np.ndarray:
    """
    Scores EEG channels based on how often the high-frequency, high-p2p
    activity exceeds a threshold in sliding windows.

    Args:
        data: Data object.
        sample_rate: Sampling rate in Hz
        window_duration: Length of each window in seconds (e.g., 0.2s)
        p2p_threshold: Minimum p2p amplitude to consider a burst

    Returns:
        score:
            Fraction of windows not exceeding p2p threshold per channel,
            shape (n_channels,), range [0, 1]
    """
    n_samples, n_channels = data.array.shape
    window_samples = int(window_duration * data.sample_rate)
    n_windows = n_samples // window_samples
    if n_windows == 0:
        return np.zeros(n_channels)

    reshaped = data[: n_windows * window_samples].array.reshape(
        n_windows, window_samples, n_channels
    )

    p2p = np.ptp(reshaped, axis=1)  # shape (n_windows, n_channels)

    burst_mask = p2p >= p2p_threshold
    return np.mean(~burst_mask, axis=0)
