import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

from zmax_datasets import settings
from zmax_datasets.utils.exceptions import (
    MultipleSleepScoringFilesFoundError,
    SleepScoringFileNotFoundError,
)

logger = logging.getLogger(__name__)

_EEG_CHANNELS: list[str] = ["EEG L", "EEG R"]
_DATA_TYPES: list[str] = _EEG_CHANNELS + [
    "dX",
    "dY",
    "dZ",
    "BATT",
    "BODY TEMP",
    "LIGHT",
    "NOISE",
    "OXY_IR_AC",
    "OXY_IR_AC_PARSED",
    "HR",
    "NASAL L",
    "NASAL R",
    "OXY_DARK_AC",
    "OXY_IR_DC",
    "OXY_R_AC",
    "OXY_R_DC",
    "ROOM C",
    "RSSI",
]
_SLEEP_SCORING_FILE_SEPARATORS: list[str] = [" ", "\t"]


@dataclass
class ZMaxRecording:
    subject_id: str
    session_id: str
    data_dir: Path
    _sleep_scoring_file: Path | None = field(default=None, repr=False, init=False)

    @property
    def sleep_scoring_file(self) -> Path | None:
        return self._sleep_scoring_file

    @sleep_scoring_file.setter
    def sleep_scoring_file(self, value: Path | None) -> None:
        if value is not None and not value.is_file():
            raise FileNotFoundError(f"Sleep scoring file {value} does not exist.")
        self._sleep_scoring_file = value

    @property
    def data_types(self) -> list[str]:
        return [
            data_type
            for data_type_file in self.data_dir.glob(
                f"*.{settings.ZMAX['data_types_file_extension']}"
            )
            if (data_type := data_type_file.stem) in _DATA_TYPES
        ]

    def __str__(self) -> str:
        return f"s{self.subject_id}_n{self.session_id}"


class ZMaxDataset(ABC):
    def __init__(
        self,
        data_dir: Path | str,
    ):
        self.data_dir = Path(data_dir)

    def get_recordings(
        self, with_sleep_scoring: bool = False
    ) -> Generator[ZMaxRecording, None, None]:
        for zmax_dir in self._zmax_dir_generator():
            subject_id, session_id = self._extract_ids_from_zmax_dir(zmax_dir)
            recording = self._create_recording(subject_id, session_id, zmax_dir)

            if with_sleep_scoring and not recording.sleep_scoring_file:
                continue

            yield recording

    @abstractmethod
    def _zmax_dir_generator(self) -> Generator[Path, None, None]: ...

    @classmethod
    @abstractmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        """
        Extract subject and session IDs from ZMax directory.

        Args:
            zmax_dir (Path): The path to the ZMax directory.

        Returns:
            tuple[str, str]:
                subject_id (str): The subject ID.
                session_id (str): The session ID.
        """
        ...

    def _create_recording(
        self, subject_id: str, session_id: str, zmax_dir: Path
    ) -> ZMaxRecording:
        recording = ZMaxRecording(subject_id, session_id, zmax_dir)
        try:
            recording.sleep_scoring_file = self._get_sleep_scoring_file(recording)
        except (
            FileNotFoundError,
            SleepScoringFileNotFoundError,
            MultipleSleepScoringFilesFoundError,
        ) as err:
            logger.info(f"Could not set the sleep scoring file for {recording}: {err}")
        return recording

    @abstractmethod
    def _get_sleep_scoring_file(self, recording: ZMaxRecording) -> Path: ...
