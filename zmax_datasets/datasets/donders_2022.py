import logging
from collections.abc import Generator
from pathlib import Path

from zmax_datasets import settings
from zmax_datasets.datasets.base import (
    ExistingFileHandlingStrategy,
    MissingDataTypeHandlingStrategy,
    ZMaxDataset,
    ZMaxRecording,
)
from zmax_datasets.utils.helpers import load_yaml_config
from zmax_datasets.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# TODO: all of these variables should be configurable
_ZMAX_DIR_PATTERN = "s*/n*/zmax/"
_SLEEP_SCORING_FILE_NAME_PATTERN = "{subject_id} {session_id}_psg.txt"


class Donders2022(ZMaxDataset):
    def _zmax_dir_generator(self) -> Generator[Path, None, None]:
        yield from self.data_dir.glob(_ZMAX_DIR_PATTERN)

    @classmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        return zmax_dir.parent.parent.name, zmax_dir.parent.name

    def _get_sleep_scoring_file(self, recording: ZMaxRecording) -> Path:
        return recording.data_dir / _SLEEP_SCORING_FILE_NAME_PATTERN.format(
            subject_id=recording.subject_id,
            session_id=recording.session_id,
        )


if __name__ == "__main__":
    config_file = settings.CONFIG_DIR / "datasets.yaml"
    setup_logging()
    config = load_yaml_config(config_file)
    dataset = Donders2022(**config["datasets"]["donders_2022"])
    dataset.to_usleep(
        out_dir=settings.DATA_DIR / "donders_2022",
        data_types=["EEG R", "EEG L"],
        data_type_labels={
            "EEG L": "F7-Fpz",
            "EEG R": "F8-Fpz",
        },
        existing_file_handling=ExistingFileHandlingStrategy.OVERWRITE,
        missing_data_type_handling=MissingDataTypeHandlingStrategy.SKIP,
    )
    # dataset.preprare_recordings()
    # print(dataset.recordings)
    # print(dataset[0])
