import logging
from pathlib import Path

from zmax_datasets.datasets.base import (
    ZMaxDataset,
    ZMaxRecording,
)

logger = logging.getLogger(__name__)


class Donders2018(ZMaxDataset):
    @classmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        return zmax_dir.parent.name.replace("_", ""), zmax_dir.name.replace("_", "")

    def _get_sleep_scoring_file(self, recording: ZMaxRecording) -> Path:
        return self._sleep_scoring_dir / self._sleep_scoring_file_pattern.format(
            subject_id=recording.subject_id,
            session_id=recording.session_id,
        )
