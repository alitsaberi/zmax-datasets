import re

import numpy as np
from loguru import logger
from scipy.signal import find_peaks

from zmax_datasets import settings
from zmax_datasets.transforms.helpers import FIRFilter
from zmax_datasets.utils.data import Data, Event

DEFAULTS = settings.EYE_MOVEMENT_DETECTION


def detect_lr_eye_movements(
    data: Data,
    left_eeg_label: str,
    right_eeg_label: str,
    accepted_pattern: str | None = None,
    min_peak_amplitude: float = DEFAULTS["min_peak_amplitude"],
    max_peak_amplitude: float = DEFAULTS["max_peak_amplitude"],
    min_peak_gap: float = DEFAULTS["min_peak_gap"],
    relative_peak_prominence: float = DEFAULTS["relative_peak_prominence"],
    min_event_duration: float = DEFAULTS["min_event_duration"],
    max_event_duration: float = DEFAULTS["max_event_duration"],
    min_event_skewness: float = DEFAULTS["min_event_skewness"],
    max_event_skewness: float = DEFAULTS["max_event_skewness"],
    max_event_gap: float = DEFAULTS["max_event_gap"],
    min_sequence_correlation: float = DEFAULTS["min_sequence_correlation"],
    min_sequence_amplitude_ratio: float = DEFAULTS["min_sequence_amplitude_ratio"],
    max_sequence_amplitude_ratio: float = DEFAULTS["max_sequence_amplitude_ratio"],
    relative_baseline: float = DEFAULTS["relative_baseline"],
    low_cutoff: float = DEFAULTS["low_cutoff"],
    high_cutoff: float = DEFAULTS["high_cutoff"],
    artifact_mask: np.ndarray | None = None,
) -> tuple[list[Event], list[Event]]:
    """Detect left/right eye movements from a pair of frontal EEG channels.

    This method is inspired by EOG rapid eye movement detection in YASA
    (`yasa.rem_detect`).

    Args:
        data: EEG data containing (at least) `left_eeg_label` and `right_eeg_label`.
        left_eeg_label: Channel name for the "left" signal.
        right_eeg_label: Channel name for the "right" signal.
        accepted_pattern:
            Optional regex pattern used to keep/discard detected sequences based on
            their label (e.g., "LRLR"). If None, no regex-based filtering is applied.
        min_peak_amplitude: Minimum peak amplitude (in the same units as `data`).
        max_peak_amplitude: Maximum peak amplitude (in the same units as `data`).
        min_peak_gap: Minimum time between neighboring peaks (seconds).
        relative_peak_prominence: Peak prominence relative to `min_peak_amplitude`.
        min_event_duration: Minimum event duration (seconds).
        max_event_duration: Maximum event duration (seconds).
        min_event_skewness:
            Minimum event skewness (\(0\) means peak is centered; negative means
            early; positive means late).
        max_event_skewness:
            Maximum event skewness (\(0\) means peak is centered; negative means
            early; positive means late).
        max_event_gap: Maximum time between neighboring events to merge (seconds).
        min_sequence_correlation:
            Minimum (absolute) negative correlation between left and right signals
            within a candidate sequence.
        min_sequence_amplitude_ratio: Minimum std ratio between left and right signals.
        max_sequence_amplitude_ratio: Maximum std ratio between left and right signals.
        relative_baseline:
            Relative baseline threshold used to determine event start/end.
        low_cutoff: Low cutoff frequency for filtering (Hz).
        high_cutoff: High cutoff frequency for filtering (Hz).
        artifact_mask:
            Optional boolean array (shape: `(data.length,)`) marking samples to exclude
            from peak detection.

    Returns:
        A tuple `(sequences, events)` where:
        - `sequences` is a list of merged multi-event sequences (labels like "LRLR").
        - `events` is a list of individual events (each labeled "L" or "R").

    Raises:
        ValueError: If `artifact_mask` does not match `(data.length,)`.
        ValueError: If required channels are not present in `data`.
    """

    if artifact_mask is not None and artifact_mask.shape != (data.length,):
        raise ValueError(
            f"Artifact mask shape {artifact_mask.shape}"
            f" does not match data length {data.length}"
        )

    expected_order = [left_eeg_label, right_eeg_label]

    if set(expected_order) - set(data.channel_names):
        raise ValueError(f"Data must have the following channels: {expected_order}")

    data = data[:, expected_order]

    filtered_data = FIRFilter(
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
    )(data)

    # Create a negative product signal to detect movements (amplitude changes)
    product_data = Data(
        array=-(
            filtered_data[:, left_eeg_label].array
            * filtered_data[:, right_eeg_label].array
        ),
        sample_rate=data.sample_rate,
        timestamps=data.timestamps,
    )

    # Create a difference signal to determine direction
    difference_data = Data(
        array=filtered_data[:, left_eeg_label].array
        - filtered_data[:, right_eeg_label].array,
        sample_rate=data.sample_rate,
        timestamps=data.timestamps,
    )

    logger.debug(
        f"Negative product data: {product_data},"
        f" Min: {product_data.array.min()},"
        f" Max: {product_data.array.max()}"
    )
    logger.debug(
        f"Difference data: {difference_data},"
        f" Min: {difference_data.array.min()},"
        f" Max: {difference_data.array.max()}"
    )

    events = _detect_events(
        product_data,
        difference_data,
        min_peak_amplitude,
        max_peak_amplitude,
        min_peak_gap,
        relative_peak_prominence,
        min_event_duration,
        max_event_duration,
        min_event_skewness,
        max_event_skewness,
        relative_baseline,
        artifact_mask,
    )

    logger.info(f"Found {len(events)} events")
    logger.debug(f"Events: {events}")

    sequences = _build_sequences(events, max_event_gap)

    logger.info(f"Found {len(sequences)} sequences")
    logger.debug(f"Sequences: {sequences}")

    sequences = _filter_sequences(
        sequences,
        filtered_data,
        min_sequence_correlation,
        min_sequence_amplitude_ratio,
        max_sequence_amplitude_ratio,
        accepted_pattern,
    )

    logger.info(f"Found {len(sequences)} valid sequences")
    logger.debug(f"Sequences: {sequences}")

    return sequences, events


def _detect_events(
    product_data: Data,
    difference_data: Data,
    min_peak_amplitude: float,
    max_peak_amplitude: float,
    min_peak_gap: float,
    relative_peak_prominence: float,
    min_event_duration: float,
    max_event_duration: float,
    min_event_skewness: float,
    max_event_skewness: float,
    relative_baseline: float,
    artifact_mask: np.ndarray | None = None,
) -> list[Event]:
    """
    Returns an ordered list of Event objects
    representing detected eye movements with opposite polarity.
    """
    product_signal = product_data.array.squeeze()
    product_height = (min_peak_amplitude**2, max_peak_amplitude**2)
    product_peaks, _ = find_peaks(
        product_signal,
        height=product_height,
        distance=int(min_peak_gap * product_data.sample_rate),
        prominence=relative_peak_prominence * product_height[0],
    )
    logger.debug(f"Found {len(product_peaks)} product peaks")

    if artifact_mask is not None:
        product_peaks = product_peaks[~artifact_mask[product_peaks]]
        logger.debug(f"Filtered {len(product_peaks)} product peaks")

    events = []
    for product_peak_idx in product_peaks:
        start_idx = product_peak_idx
        while (
            start_idx > 0
            and product_signal[start_idx] > relative_baseline * product_height[0]
        ):
            start_idx -= 1

        end_idx = product_peak_idx
        while (
            end_idx < len(product_signal) - 1
            and product_signal[end_idx] > relative_baseline * product_height[0]
        ):
            end_idx += 1

        event_width = end_idx - start_idx
        if event_width == 0:
            logger.debug(
                f"Skipping peak at {product_data.timestamps[product_peak_idx]/1e9:.2f}s"
                f" due to event width {event_width}"
            )
            continue

        # Calculate skewness
        # 0 means peak is in the middle
        # negative means peak is early
        # positive means peak is late
        skewness = ((product_peak_idx - start_idx) / event_width) - 0.5
        duration = event_width / product_data.sample_rate

        if not (min_event_skewness <= skewness <= max_event_skewness):
            logger.debug(
                f"Skipping peak at {product_data.timestamps[product_peak_idx]/1e9:.2f}s"
                f" due to skewness {skewness:.2f}"
            )
            continue

        if not (min_event_duration <= duration <= max_event_duration):
            logger.debug(
                f"Skipping peak at {product_data.timestamps[product_peak_idx]/1e9:.2f}s"
                f" due to duration {duration:.2f}s"
            )
            continue

        events.append(
            Event(
                label=DEFAULTS[
                    "left_label"
                    if difference_data.array.squeeze()[product_peak_idx] > 0
                    else "right_label"
                ],
                start_time=product_data.timestamps[start_idx],
                end_time=product_data.timestamps[end_idx],
            )
        )

    return events


def _build_sequences(events: list[Event], max_event_gap: float) -> list[Event]:
    if not events:
        logger.debug("events is empty")
        return []

    sequences = []
    current_sequence = [events[0]]

    for event in events[1:]:
        if abs(event.start_time - current_sequence[-1].end_time) <= max_event_gap * 1e9:
            current_sequence.append(event)
        else:
            sequences.append(_merge_events(current_sequence))
            current_sequence = [event]

    # Add the last sequence
    sequences.append(_merge_events(current_sequence))

    return sequences


def _merge_events(events: list[Event]) -> Event:
    return Event(
        label="".join([event.label for event in events]),
        start_time=events[0].start_time,
        end_time=events[-1].end_time,
    )


def _filter_sequences(
    sequences: list[Event],
    filtered_data: Data,
    min_sequence_correlation: float,
    min_sequence_amplitude_ratio: float,
    max_sequence_amplitude_ratio: float,
    accepted_pattern: str | None,
) -> list[Event]:
    valid_sequences = []
    for sequence in sequences:
        if accepted_pattern is not None and not re.match(
            accepted_pattern, sequence.label
        ):
            logger.info(
                f"Sequence {sequence.label}"
                f" at {sequence.start_time}-{sequence.end_time}"
                f" discarded due to pattern {accepted_pattern}"
            )
            continue

        mask = (filtered_data.timestamps >= sequence.start_time) & (
            filtered_data.timestamps <= sequence.end_time
        )
        masked_data = filtered_data.array[mask]

        left = masked_data[:, 0]
        right = masked_data[:, 1]

        correlation = np.corrcoef(left.squeeze(), right.squeeze())[0, 1]

        if correlation > -min_sequence_correlation:
            logger.info(
                f"Sequence {sequence.label}"
                f" at {sequence.start_time}-{sequence.end_time} "
                f"discarded due to low negative correlation: {correlation}"
            )
            continue

        amplitude_ratio = np.std(left) / (np.std(right) + np.finfo(float).eps)
        if (
            min_sequence_amplitude_ratio > amplitude_ratio
            or amplitude_ratio > max_sequence_amplitude_ratio
        ):
            logger.info(
                f"Sequence {sequence.label}"
                f" at {sequence.start_time}-{sequence.end_time}"
                f" discarded due to amplitude ratio: {amplitude_ratio}"
            )
            continue

        valid_sequences.append(sequence)

    return valid_sequences
