import logging
from collections.abc import Generator
from pathlib import Path

import pandas as pd

from zmax_datasets import settings
from zmax_datasets.datasets.base import (
    ZMaxDataset,
    ZMaxRecording,
)
from zmax_datasets.exports.usleep import (
    ErrorHandling,
    ExistingFileHandling,
    USleepExportStrategy,
)
from zmax_datasets.utils.exceptions import (
    MultipleSleepScoringFilesFoundError,
    SleepScoringFileNotFoundError,
)
from zmax_datasets.utils.helpers import load_yaml_config
from zmax_datasets.utils.logger import setup_logging

logger = logging.getLogger(__name__)


# TODO: all of these variables should be configurable
_ZMAX_DIR_PATTERN = "data/PSG/Zmax/original_recordings/*/*/"
_SCORING_DIR = "Organized QS data/All_in_one_scoring_for_ZMax/"
_SCORING_MAPPING_FILE = Path(__file__).parent / "qs_scoring_files.csv"
_SUBJECT_ID = "s1"
_USLEEP_HYPNOGRAM_MAPPING: dict[int, str] = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    5: "REM",
    -1: "UNKNOWN",
}


class QS(ZMaxDataset):
    def __init__(
        self,
        data_dir: Path | str,
        zmax_dir_pattern: str = _ZMAX_DIR_PATTERN,
        hypnogram_mapping: dict[int, str] = _USLEEP_HYPNOGRAM_MAPPING,
    ):
        super().__init__(data_dir, zmax_dir_pattern, hypnogram_mapping)
        self._scoring_mapping = self._load_scoring_mapping()

    def _load_scoring_mapping(self) -> pd.DataFrame:
        return pd.read_csv(
            _SCORING_MAPPING_FILE,
            names=["session_id", "scoring_file"],  # TODO: should not be hardcoded
        )

    def _zmax_dir_generator(self) -> Generator[Path, None, None]:
        yield from self.data_dir.glob(_ZMAX_DIR_PATTERN)

    @classmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        return _SUBJECT_ID, zmax_dir.name

    def _get_sleep_scoring_file(self, recording: ZMaxRecording) -> Path:
        matching_rows = self._scoring_mapping[
            self._scoring_mapping["session_id"] == recording.session_id
        ]

        if matching_rows.empty:
            raise SleepScoringFileNotFoundError(
                f"No scoring file found for {recording}."
            )

        if (scoring_files_count := len(matching_rows)) > 1:
            raise MultipleSleepScoringFilesFoundError(
                f"Multiple scoring files ({scoring_files_count}) found for {recording}."
            )

        return self.data_dir / _SCORING_DIR / matching_rows["scoring_file"].iloc[0]


if __name__ == "__main__":
    setup_logging()
    config_file = settings.CONFIG_DIR / "datasets.yaml"
    config = load_yaml_config(config_file)
    dataset = QS(**config["datasets"]["qs"])
    export_strategy = USleepExportStrategy(
        data_types=["EEG R", "EEG L"],
        data_type_labels={
            "EEG L": "F7-Fpz",
            "EEG R": "F8-Fpz",
        },
        existing_file_handling=ExistingFileHandling.OVERWRITE,
        error_handling=ErrorHandling.SKIP,
    )
    export_strategy.export(dataset, Path("data/qs"))
