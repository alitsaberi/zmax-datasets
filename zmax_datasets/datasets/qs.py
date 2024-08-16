import logging
from collections.abc import Generator
from pathlib import Path

import pandas as pd

from zmax_datasets import settings
from zmax_datasets.datasets.zmax import HandlingStrategy, ZMaxDataset, ZMaxRecording
from zmax_datasets.utils.helpers import load_yaml_config
from zmax_datasets.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class QS(ZMaxDataset):
    _ZMAX_DIR_PATTERN = "data/PSG/Zmax/original_recordings/**/*"
    _SCORING_DIR = "Organized QS data/All_in_one_scoring_for_ZMax/"
    _SCORING_MAPPING_FILE = Path(__file__).parent / "qs_scoring_files.csv"
    _SUBJECT_ID = "1"
    _USLEEP_HYPNOGRAM_MAPPING: dict[int, str] = {
        0: "W",
        1: "N1",
        2: "N2",
        3: "N3",
        5: "REM",
        -1: "UNKNOWN",
    }

    def __init__(self, data_dir: Path | str):
        super().__init__(data_dir)
        self._scoring_mapping = self._load_scoring_mapping()

    def _load_scoring_mapping(self) -> pd.DataFrame:
        return pd.read_csv(
            self._SCORING_MAPPING_FILE, names=["session_id", "scoring_file"]
        )

    def get_recordings(self) -> Generator[ZMaxRecording, None, None]:
        for zmax_dir in self.data_dir.glob(f"{self._ZMAX_DIR_PATTERN}"):
            if (recording := self._process_zmax_dir(zmax_dir)) is not None:
                yield recording

    @classmethod
    def _extract_ids_from_zmax_dir(
        cls, zmax_dir: Path
    ) -> tuple[int | None, int | None]:
        return cls._SUBJECT_ID, zmax_dir.name

    def _get_sleep_scores_file(
        self, zmax_dir: Path, subject_id: int, session_id: int
    ) -> Path | None:
        matching_row = self._scoring_mapping[
            self._scoring_mapping["session_id"] == session_id
        ]
        return (
            self.data_dir / self._SCORING_DIR / matching_row["scoring_file"].iloc[0]
            if not matching_row.empty
            else None
        )


if __name__ == "__main__":
    config_file = settings.CONFIG_DIR / "datasets.yaml"
    setup_logging()
    config = load_yaml_config(config_file)
    dataset = QS(**config["datasets"]["qs"])
    dataset.to_usleep(
        out_dir=settings.DATA_DIR / "qs",
        data_types=["EEG R", "EEG L"],
        data_type_labels={
            "EEG L": "F7-Fpz",
            "EEG R": "F8-Fpz",
        },
        existing_file_handling=HandlingStrategy.SKIP,
        missing_data_type_handling=HandlingStrategy.SKIP,
    )
