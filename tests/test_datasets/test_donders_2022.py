import re
from pathlib import Path

import pytest

from zmax_real_time.datasets.donders_2022 import Donders2022
from zmax_real_time.utils import load_yaml_config


@pytest.fixture(scope="module")
def mock_data_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("data")
    data_dir = tmp_path / "mock_data"
    data_dir.mkdir()
    (data_dir / "s1" / "n1" / "zmax").mkdir(parents=True)
    (data_dir / "s1" / "n1" / "somno").mkdir()
    (data_dir / "s2" / "n1" / "zmax").mkdir(parents=True)
    (data_dir / "s2" / "n1" / "somno").mkdir()
    (data_dir / "s3" / "n1" / "zmax").mkdir(parents=True)
    (data_dir / "s3" / "n1" / "somno").mkdir()
    return data_dir


def test_dataset_initialization(mock_data_dir):
    dataset = Donders2022(mock_data_dir)
    assert len(dataset.session_dirs) == 3


def test_dataset_with_selected_subjects(mock_data_dir):
    dataset = Donders2022(mock_data_dir, selected_subjects=[1, 2])
    assert len(dataset.session_dirs) == 2


def test_dataset_with_excluded_subjects(mock_data_dir):
    dataset = Donders2022(mock_data_dir, excluded_subjects=[3])
    assert len(dataset.session_dirs) == 2


def test_dataset_with_custom_regex(mock_data_dir):
    dataset = Donders2022(mock_data_dir, subject_regex=r"s(?P<id>\d{3})")
    assert len(dataset.session_dirs) == 0


def test_dataset_with_invalid_regex(mock_data_dir):
    with pytest.raises(
        ValueError, match="subject_regex must include a group named 'id'"
    ):
        Donders2022(mock_data_dir, subject_regex=r"s\d+")


def test_invalid_directory_structure(mock_data_dir):
    # Create an invalid directory
    (mock_data_dir / "invalid_dir").mkdir()
    dataset = Donders2022(mock_data_dir)
    assert len(dataset.session_dirs) == 3  # Should ignore the invalid directory


def test_missing_zmax_directory(mock_data_dir):
    (mock_data_dir / "s1" / "n1" / "zmax").rmdir()
    dataset = Donders2022(mock_data_dir)
    assert len(dataset.session_dirs) == 2


def test_missing_psg_directory(mock_data_dir):
    (mock_data_dir / "s1" / "n1" / "somno").rmdir()
    dataset = Donders2022(mock_data_dir)
    assert len(dataset.session_dirs) == 2


def test_get_entity_id():
    assert Donders2022._get_entity_id(Path("s1"), re.compile(r"s(?P<id>\d+)")) == 1
    assert (
        Donders2022._get_entity_id(Path("invalid"), re.compile(r"s(?P<id>\d+)")) is None
    )


def test_is_valid_entity_dir(mock_data_dir):
    dataset = Donders2022(mock_data_dir)
    assert dataset._is_valid_entity_dir(
        mock_data_dir / "s1", Donders2022.EntityType.SUBJECT
    )
    assert not dataset._is_valid_entity_dir(
        mock_data_dir / "invalid", Donders2022.EntityType.SUBJECT
    )


def test_config_matches_dataset_parameters():
    config_path = Path("configs/donders_2022_config.yaml")
    config = load_yaml_config(config_path)
    dataset = Donders2022(**config)

    assert dataset.subject_regex.pattern == config["subject_regex"]
    assert dataset.session_regex.pattern == config["session_regex"]
    assert dataset.zmax_dir_name == config["zmax_dir_name"]
    assert dataset.psg_dir_name == config["psg_dir_name"]

    if "selected_subjects" in config:
        assert dataset.selected_subjects == config["selected_subjects"]
    if "selected_sessions" in config:
        assert dataset.selected_sessions == config["selected_sessions"]
    if "excluded_subjects" in config:
        assert dataset.excluded_subjects == config["excluded_subjects"]
    if "excluded_sessions" in config:
        assert dataset.excluded_sessions == config["excluded_sessions"]
