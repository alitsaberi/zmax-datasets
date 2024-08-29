import numpy as np
import pandas as pd
import pytest

from zmax_datasets.datasets.donders_2022 import Donders2022
from zmax_datasets.utils.exceptions import SleepScoringReadError


@pytest.fixture(scope="module")
def mock_data_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("data")
    data_dir = tmp_path / "mock_data"
    data_dir.mkdir()
    for subject_id in range(1, 4):
        for session_id in range(1, 4):
            recording_dir = (
                data_dir
                / f"s{subject_id}"
                / f"n{session_id}"
                / Donders2022._ZMAX_DIR_NAME
            )
            recording_dir.mkdir(parents=True)
            (
                recording_dir
                / Donders2022._MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN.format(
                    subject_id=subject_id, session_id=session_id
                )
            ).touch()

    return data_dir


@pytest.fixture
def mock_donders_dataset(mock_data_dir):
    return Donders2022(mock_data_dir)


def test_read_hypnogram(tmp_path):
    hypnogram_file = tmp_path / "mock_hypnogram.txt"
    sleep_stages = np.array([0, 1, 2, 3, 4])
    mock_data = pd.DataFrame({"sleep_stage": sleep_stages, "arousal": [0, 1, 0, 1, 0]})
    mock_data.to_csv(hypnogram_file, sep=" ", index=False, header=False)
    result = Donders2022._read_hypnogram(hypnogram_file)

    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, sleep_stages)

    # Test with an empty file
    empty_file = tmp_path / "empty_hypnogram.txt"
    empty_file.touch()
    empty_result = Donders2022._read_hypnogram(empty_file)
    assert len(empty_result) == 0

    # Test with a file containing non-integer values (should raise an error)
    invalid_file = tmp_path / "invalid_hypnogram.txt"
    pd.DataFrame({"sleep_stage": [0, 1, "a", 3, 4], "arousal": [0, 1, 0, 1, 0]}).to_csv(
        invalid_file, sep=" ", index=False, header=False
    )
    with pytest.raises(SleepScoringReadError):
        Donders2022._read_hypnogram(invalid_file)

    # Test with a file containing tab-separated columns
    tab_separated_file = tmp_path / "tab_separated_hypnogram.txt"
    tab_separated_stages = np.array([0, 1, 2, 3, 4, 5])
    pd.DataFrame(
        {"sleep_stage": tab_separated_stages, "arousal": [0, 1, 0, 1, 0, 1]}
    ).to_csv(tab_separated_file, sep="\t", index=False, header=False)
    tab_separated_result = Donders2022._read_hypnogram(tab_separated_file)
    assert isinstance(tab_separated_result, np.ndarray)
    assert tab_separated_result.shape == (6,)
    assert tab_separated_result.dtype == np.int64
    np.testing.assert_array_equal(tab_separated_result, tab_separated_stages)
