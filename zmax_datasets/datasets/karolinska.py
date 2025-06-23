import logging
from pathlib import Path

from zmax_datasets.datasets.zmax import (
    Dataset,
    Recording,
)

logger = logging.getLogger(__name__)


class Karolinska(Dataset):
    @classmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        return zmax_dir.parent.parent.name, zmax_dir.name

    def _get_sleep_scoring_file(self, recording: Recording) -> Path:
        return self._sleep_scoring_dir / self._sleep_scoring_file_pattern.format(
            subject_id=recording.subject_id.replace("SNZ_", ""),
            session_id=recording.session_id,
        )
