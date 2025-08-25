import heartpy as hp
import numpy as np
from loguru import logger
from scipy.interpolate import interp1d

# Constants
MIN_RR_VALUE = 300  # ms
MAX_RR_VALUE = 2000  # ms
MAX_INVALID_FRACTION = 0.2
MAX_DIFF_THRESHOLD = 300  # ms
MIN_UNIQUE_VALUES = 3


def extract_ibi_from_peaks(
    peaks: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract IBI from binary peaks signal.

    Args:
        peaks: Binary signal where 1 indicates peaks and 0 indicates non-peaks
        sample_rate: Signal sampling rate in Hz

    Returns:
        np.ndarray: Inter-beat intervals in milliseconds
    """
    if peaks.ndim != 1:
        raise ValueError("Peaks signal must be a 1D array.")

    # Find indices where peaks occur (where signal is 1)
    peak_indices = np.where(peaks == 1)[0]

    if len(peak_indices) < 2:
        logger.warning("Not enough peaks to calculate IBI.")
        return np.array([])

    # Calculate time differences between consecutive peaks in milliseconds
    ibi = np.diff(peak_indices) * (1000 / sample_rate)
    ibi_times = peak_indices[1:] / sample_rate

    return ibi, ibi_times


def detect_rr(signal, sample_rate, **kwargs):
    try:
        wd, _ = hp.process(signal, sample_rate, high_precision=True, **kwargs)
        peaklist = np.array(wd["peaklist"])[wd["binary_peaklist"] == 1]
        times = (peaklist / sample_rate) * 1000  # convert to ms
        rr_values = times[1:] - times[:-1]
        rr_times = (times[1:] + times[:-1]) / 2.0
        return rr_times, rr_values
    except Exception as e:
        logger.warning(f"Peak detection failed: {e}")
        return None, None


def clean_rr(rr_times, rr_values):
    mask = (rr_values >= MIN_RR_VALUE) & (rr_values <= MAX_RR_VALUE)
    return rr_times[mask], rr_values[mask]


def interpolate_rr(rr_times, rr_values, duration, target_rate):
    if len(rr_values) < 2:
        return np.zeros(int(duration * target_rate))

    interp_func = interp1d(
        rr_times,
        rr_values,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )
    target_times = np.linspace(
        0, duration * 1000, int(duration * target_rate), endpoint=False
    )
    ibi = interp_func(target_times)
    ibi = np.clip(ibi, MIN_RR_VALUE, MAX_RR_VALUE)
    return ibi


def is_segment_invalid_by_value_criteria(
    ibi,
    min_rr=MIN_RR_VALUE,
    max_rr=MAX_RR_VALUE,
    max_invalid_fraction=MAX_INVALID_FRACTION,
):
    invalid = np.sum((ibi == min_rr) | (ibi == max_rr))
    return invalid / len(ibi) > max_invalid_fraction


def is_segment_too_noisy(ibi, max_diff=MAX_DIFF_THRESHOLD):
    return np.any(np.abs(np.diff(ibi)) > max_diff)


def is_segment_too_flat(ibi, min_unique=MIN_UNIQUE_VALUES):
    return len(np.unique(ibi)) < min_unique


def extract_ibi(
    bvp: np.ndarray,
    sample_rate: float,
    segment_duration: int,
    target_rate: float,
    lookaround: int = 1,  # number of extra seconds before and after
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract interpolated IBI with regular sample rate from a BVP signal.

    Args:
        bvp: Raw BVP signal.
        sample_rate: BVP sampling rate.
        segment_duration: Segment duration in seconds.
        target_rate: Output IBI sampling rate (Hz).
        lookaround: Extra seconds before/after segment for peak detection.
        **kwargs: Passed to heartpy.process().

    Returns:
        ibi (np.ndarray): IBI signal with uniform sampling.
        usability (np.ndarray): Binary mask (1 = unusable, 0 = usable).
    """
    seg_len = int(segment_duration * sample_rate)
    look_len = int(lookaround * sample_rate)
    total_segments = len(bvp) // seg_len
    logger.info(f"Extracting {total_segments} segments")

    ibi_segments = []
    usability_mask = []

    for i in range(total_segments):
        start = max(0, i * seg_len - look_len)
        end = min(len(bvp), (i + 1) * seg_len + look_len)
        segment = bvp[start:end]

        rr_t, rr_v = detect_rr(segment, sample_rate, **kwargs)
        time_shift_ms = max(0, look_len / sample_rate * 1000)

        if rr_t is not None and rr_v is not None:
            rr_t, rr_v = clean_rr(rr_t, rr_v)
            rr_t -= time_shift_ms  # align to central segment
            ibi_segment = interpolate_rr(rr_t, rr_v, segment_duration, target_rate)

            unusable = (
                np.all(ibi_segment == 0)
                or is_segment_invalid_by_value_criteria(ibi_segment)
                or is_segment_too_noisy(ibi_segment)
                or is_segment_too_flat(ibi_segment)
            )
        else:
            ibi_segment = np.zeros(int(segment_duration * target_rate))
            unusable = True

        ibi_segments.append(ibi_segment)
        usability_mask.append(int(unusable))

        logger.debug(
            f"Segment {i+1}/{total_segments}:"
            f" unusable={unusable},"
            f" mean IBI={ibi_segment.mean():.2f}"
        )

    return np.concatenate(ibi_segments), np.array(usability_mask, dtype=int)
