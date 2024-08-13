import pandas as pd
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
            (recording_dir / f"s{subject_id} n{session_id}_psg.txt").touch()

    return data_dir


def test_dataset_initialization(mock_data_dir):
    dataset = Donders2022(mock_data_dir)
    assert len(dataset) == 9
    assert isinstance(dataset.recordings, pd.DataFrame)
    assert len(dataset.recordings) == 9
    assert set(dataset.recordings.columns) == {
        "subject_id",
        "session_id",
        "manual_sleep_scores_file",
    }


def test_dataset_with_selected_subjects(mock_data_dir):
    dataset = Donders2022(mock_data_dir, selected_subjects=[1, 2])
    assert len(dataset) == 6


def test_dataset_with_excluded_subjects(mock_data_dir):
    dataset = Donders2022(mock_data_dir, excluded_subjects=[3])
    assert len(dataset) == 6


def test_dataset_with_selected_sessions(mock_data_dir):
    dataset = Donders2022(mock_data_dir, selected_sessions=[1])
    assert len(dataset) == 3


def test_dataset_with_excluded_sessions(mock_data_dir):
    dataset = Donders2022(mock_data_dir, excluded_sessions=[2])
    assert len(dataset) == 6


def test_dataset_with_excluded_sessions_for_subjects(mock_data_dir):
    dataset = Donders2022(mock_data_dir, excluded_sessions_for_subjects={1: [1], 2: []})
    assert len(dataset) == 8


def test_extract_ids_from_filename():
    subject_id, session_id = Donders2022._extract_ids_from_file_name("s3 n8_psg.txt")
    assert subject_id == 3
    assert session_id == 8


def test_extract_ids_from_invalid_filename():
    subject_id, session_id = Donders2022._extract_ids_from_file_name(
        "invalid_filename.txt"
    )
    assert subject_id is None
    assert session_id is None
