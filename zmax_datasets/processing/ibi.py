import heartpy as hp
import numpy as np
from heartpy.datautils import outliers_iqr_method
from loguru import logger
from scipy.interpolate import interp1d

GLOBAL_RANGE_THRESHOLD_FACTOR = 0.5
MAX_VALUE_THRESHOLD_FACTOR = 0.9
MIN_VALUE_THRESHOLD_FACTOR = 0.1
MIN_RR_VALUE = 300
MAX_RR_VALUE = 2000


def is_usable(
    segment: np.ndarray, max_value: float, min_value: float, global_range: float
) -> bool:
    segment_range = np.max(segment) - np.min(segment)
    return not (
        (segment_range >= (GLOBAL_RANGE_THRESHOLD_FACTOR * global_range))
        or (np.max(segment) >= MAX_VALUE_THRESHOLD_FACTOR * max_value)
        or (np.min(segment) <= min_value + (MIN_VALUE_THRESHOLD_FACTOR * min_value))
    )


def process_segment(
    segment: np.ndarray,
    sample_rate: float,
    duration: int,
    target_sample_rate: float,
    **kwargs,
) -> np.ndarray:
    """
    Processes a segment of the heart signal.
    """
    try:
        wd, _ = hp.process(
            segment,
            sample_rate,
            high_precision=True,
            **kwargs,
        )

        beat_indices = np.array(wd["peaklist"])[wd["binary_peaklist"] == 1]
        beat_times = (beat_indices / sample_rate) * 1000  # convert to ms

        logger.debug(f"Beat indices ({len(beat_indices)}): {beat_indices}")
        logger.debug(f"Beat times ({len(beat_times)}): {beat_times}")

        rr_values = beat_times[1:] - beat_times[:-1]
        rr_times = (beat_times[1:] + beat_times[:-1]) / 2.0
        logger.debug(f"RR times ({len(rr_times)}): {rr_times}")
        logger.debug(f"RR values with outliers ({len(rr_values)}): {rr_values}")

        rr_values, _ = outliers_iqr_method(rr_values)
        rr_values = np.array(rr_values)
        logger.debug(f"RR values without outliers ({len(rr_values)}): {rr_values}")

        valid_rr_values = (rr_values >= MIN_RR_VALUE) & (rr_values <= MAX_RR_VALUE)
        if not all(valid_rr_values):
            logger.warning(f"Invalid RR values detected: {rr_values[~valid_rr_values]}")

        rr_times = rr_times[valid_rr_values]
        rr_values = rr_values[valid_rr_values]
        logger.debug(f"RR times without outliers ({len(rr_times)}): {rr_times}")
        logger.debug(f"RR values without outliers ({len(rr_values)}): {rr_values}")

        regular_times = np.linspace(
            0, duration * 1000, int(duration * target_sample_rate), endpoint=False
        )

        interp_func = interp1d(
            rr_times, rr_values, kind="linear", fill_value="extrapolate"
        )
        ibi_signal = interp_func(regular_times)
        logger.debug(f"IBI signal ({len(ibi_signal)}): {ibi_signal}")

        valid_ibi_values = (ibi_signal >= MIN_RR_VALUE) & (ibi_signal <= MAX_RR_VALUE)
        if not all(valid_ibi_values):
            logger.warning(
                f"Invalid IBI values detected: {ibi_signal[~valid_ibi_values]}"
            )

        ibi_signal = np.clip(ibi_signal, MIN_RR_VALUE, MAX_RR_VALUE)
        logger.debug(f"IBI signal clipped ({len(ibi_signal)}): {ibi_signal}")

        return ibi_signal
    except hp.exceptions.BadSignalWarning as e:
        logger.info(f"Bad signal warning: {e}")
    except Exception as e:
        logger.error(f"Error processing segment: {e}")

    return np.zeros(int(duration * target_sample_rate))


def extract_ibi(
    heart_signal: np.ndarray,
    sample_rate: float,
    segment_duration: int,
    target_sample_rate: float,
    **kwargs,
) -> tuple[np.ndarray, list[bool]]:
    """
    Extracts Inter-Beat Intervals (IBI) from the heart signal.

    Args:
        heart_signal (np.ndarray): The heart signal.
        sample_rate (float): The sample rate of the heart signal in Hz.
        segment_duration (int): The duration of each segment in seconds.
        target_sample_rate (float): The target sample rate of the IBI signal in Hz.
        reject_outliers (bool): Whether to reject outliers from the IBI signal.
        outlier_detection_method (str): The method to use for outlier detection.
        **kwargs: Additional arguments to pass to the heartpy process function.

    Returns:
        tuple[np.ndarray, list[bool]]: The IBI signal and the usability labels.
    """

    max_value = np.max(heart_signal)
    min_value = np.min(heart_signal)
    global_range = max_value - min_value

    segment_size = int(segment_duration * sample_rate)
    n_segments = len(heart_signal) // segment_size
    logger.info(f"Processing {n_segments} segments")

    ibi_signals = []
    usability_labels = []

    for i in range(n_segments):
        segment = heart_signal[i * segment_size : (i + 1) * segment_size]

        is_usable_segment = is_usable(segment, max_value, min_value, global_range)

        if is_usable_segment:
            ibi_signal = process_segment(
                segment,
                sample_rate,
                segment_duration,
                target_sample_rate,
                **kwargs,
            )
        else:
            ibi_signal = np.zeros(int(segment_duration * target_sample_rate))

        usability_label = not is_usable_segment or np.all(ibi_signal == 0)

        logger.debug(
            f"Processed segment {i+1}/{n_segments}."
            f" Usability label: {usability_label}"
        )

        ibi_signals.append(ibi_signal)
        usability_labels.append(usability_label)

    return np.concatenate(ibi_signals), np.array(usability_labels, dtype=int)
