import numpy as np
import pytest

from zmax_datasets.processing.eye_movement_detection import detect_lr_eye_movements
from zmax_datasets.utils.data import Data


def _make_two_channel_data(
    *,
    left_channel: str = "Fp1",
    right_channel: str = "Fp2",
    sample_rate: float = 100.0,
    n_samples: int = 500,
    pulse_center_s: float = 2.0,
    pulse_sigma_s: float = 0.05,
    pulse_amplitude: float = 300.0,
) -> Data:
    t = np.arange(n_samples) / sample_rate
    pulse = pulse_amplitude * np.exp(
        -0.5 * ((t - pulse_center_s) / pulse_sigma_s) ** 2
    )

    left = pulse
    right = -pulse

    array = np.stack([left, right], axis=1)
    return Data(
        array=array,
        sample_rate=sample_rate,
        channel_names=[left_channel, right_channel],
    )


def test_detect_lr_eye_movements_raises_on_missing_channels() -> None:
    data = Data(
        array=np.zeros((100, 2)),
        sample_rate=100.0,
        channel_names=["A", "B"],
    )

    with pytest.raises(ValueError, match=r"Data must have the following channels"):
        detect_lr_eye_movements(data, left_eeg_label="Fp1", right_eeg_label="Fp2")


def test_detect_lr_eye_movements_raises_on_artifact_mask_shape_mismatch() -> None:
    data = _make_two_channel_data(n_samples=200)
    artifact_mask = np.zeros((10,), dtype=bool)

    with pytest.raises(ValueError, match=r"Artifact mask shape"):
        detect_lr_eye_movements(
            data,
            left_eeg_label="Fp1",
            right_eeg_label="Fp2",
            artifact_mask=artifact_mask,
        )


def test_detect_lr_eye_movements_can_filter_sequences_by_regex() -> None:
    data = _make_two_channel_data()

    sequences, events = detect_lr_eye_movements(
        data,
        left_eeg_label="Fp1",
        right_eeg_label="Fp2",
        accepted_pattern=None,
    )
    assert len(events) >= 1
    assert len(sequences) >= 1

    sequences_filtered, events_filtered = detect_lr_eye_movements(
        data,
        left_eeg_label="Fp1",
        right_eeg_label="Fp2",
        accepted_pattern=r"^$",
    )
    assert len(events_filtered) >= 1
    assert sequences_filtered == []

