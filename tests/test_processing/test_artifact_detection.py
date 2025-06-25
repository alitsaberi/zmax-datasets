import numpy as np
import pytest

from zmax_datasets.processing.artifact_detection import get_usability_scores, load_model


@pytest.fixture
def mock_data():
    # Create mock data for testing
    data = np.random.rand(3 * 10 * 256, 3)  # 1000 samples, 3 channels
    sample_rate = 256.0  # 256 Hz sample rate
    return data, sample_rate


def test_get_usability_scores(mock_data):
    data, sample_rate = mock_data
    eeg_left_channel_index = 0
    eeg_right_channel_index = 1
    movement_channel_index = 2
    model = load_model()

    usability_scores, data, epoch_length = get_usability_scores(
        data,
        sample_rate,
        model,
        eeg_left_channel_index,
        eeg_right_channel_index,
        movement_channel_index,
    )

    assert usability_scores.shape == (3, 2)
    assert data.shape == (3 * 10 * 256, 3)
    assert epoch_length == sample_rate * 10
