import pytest

from zmax_datasets.datasets.donders_2022 import Donders2022


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
