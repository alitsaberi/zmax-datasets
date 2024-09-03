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


_USLEEP_HYPNOGRAM_MAPPING: dict[int, str] = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    5: "REM",
    -1: "UNKNOWN",
}


class Donders2018(ZMaxDataset):
    def __init__(
        self,
        data_dir: Path | str,
        zmax_dir_pattern: str,
        sleep_scoring_dir: Path | str | None = None,
        sleep_scoring_file_pattern: str | None = None,
        hypnogram_mapping: dict[int, str] = _USLEEP_HYPNOGRAM_MAPPING,
    ):
        super().__init__(
            data_dir,
            zmax_dir_pattern,
            sleep_scoring_dir,
            sleep_scoring_file_pattern,
            hypnogram_mapping,
        )

    @classmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        return zmax_dir.parent.name.replace("_", ""), zmax_dir.name.replace("_", "")

    def _get_sleep_scoring_file(self, recording: ZMaxRecording) -> Path:
        return self._sleep_scoring_dir / self._sleep_scoring_file_pattern.format(
            subject_id=recording.subject_id,
            session_id=recording.session_id,
        )


if __name__ == "__main__":
    setup_logging()
    config_file = settings.CONFIG_DIR / "datasets.yaml"
    config = load_yaml_config(config_file)
    dataset = Donders2018(**config["datasets"]["donders_2018"])
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
