import logging
from pathlib import Path

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
from zmax_datasets.utils.helpers import load_yaml_config
from zmax_datasets.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# TODO: all of these variables should be configurable
_ZMAX_DIR_PATTERN = "Donders2018/Zmax_Data/P*/night*"
_SLEEP_SCORING_FILE_NAME_PATTERN = (
    "AdaptedScorings/All_in_one_scoring_for_ZMax/"
    "{subject_id} {session_id}_all_in_one_ZMax.txt"
)


class Donders2022(ZMaxDataset):
    def __init__(
        self,
        data_dir: Path | str,
        zmax_dir_pattern: str = _ZMAX_DIR_PATTERN,
        hypnogram_mapping: dict[int, str] = settings.USLEEP[
            "default_hypnogram_mapping"
        ],
    ):
        super().__init__(data_dir, zmax_dir_pattern, hypnogram_mapping)

    @classmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        return zmax_dir.parent.name.replace("_", ""), zmax_dir.name.replace("_", "")

    def _get_sleep_scoring_file(self, recording: ZMaxRecording) -> Path:
        return self.data_dir / _SLEEP_SCORING_FILE_NAME_PATTERN.format(
            subject_id=recording.subject_id.replace("_", ""),
            session_id=recording.session_id.replace("_", ""),
        )


if __name__ == "__main__":
    setup_logging()
    config_file = settings.CONFIG_DIR / "datasets.yaml"
    config = load_yaml_config(config_file)
    dataset = Donders2022(**config["datasets"]["donders_2018"])
    export_strategy = USleepExportStrategy(
        data_types=["EEG R", "EEG L"],
        data_type_labels={
            "EEG L": "F7-Fpz",
            "EEG R": "F8-Fpz",
        },
        existing_file_handling=ExistingFileHandling.OVERWRITE,
        error_handling=ErrorHandling.SKIP,
    )
    export_strategy.export(dataset, Path("data/donders_2018"))
