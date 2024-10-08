import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from zmax_datasets import settings
from zmax_datasets.datasets.utils import mapper
from zmax_datasets.utils.exceptions import (
    MultipleSleepScoringFilesFoundError,
    SleepScoringFileNotFoundError,
    SleepScoringReadError,
)

logger = logging.getLogger(__name__)

EEG_CHANNELS: list[str] = ["EEG L", "EEG R"]
_DATA_TYPES: list[str] = EEG_CHANNELS + [
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
_SLEEP_SCORING_FILE_COLUMNS = ["sleep_stage", "arousal"]


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
        return f"{self.subject_id}_{self.session_id}"

    def read_raw_data(self, data_type: str) -> mne.io.Raw:
        """Extract and return raw data from an EDF file for a specific data type.

        This method reads an EDF file corresponding to the given data type
        and returns the raw MNE object without further processing.

        Args:
            data_type (str): The type of data to extract (e.g., 'EEG L', 'EEG R').

        Returns:
            mne.io.Raw: The raw MNE object containing the extracted data.

        Raises:
            FileNotFoundError: If the EDF file for the specified data type is not found.
        """
        logger.info(f"Extracting {data_type}")
        data_type_file = (
            self.data_dir / f"{data_type}.{settings.ZMAX['data_types_file_extension']}"
        )
        raw = mne.io.read_raw_edf(data_type_file, preload=False)
        logger.debug(f"Channels: {raw.info['chs']}")

        return raw

    def read_sleep_scoring(self) -> pd.DataFrame:
        for separator in _SLEEP_SCORING_FILE_SEPARATORS:
            try:
                return pd.read_csv(
                    self.sleep_scoring_file,
                    sep=separator,
                    names=_SLEEP_SCORING_FILE_COLUMNS,
                    dtype=int,
                )
            except ValueError as e:
                logger.debug(
                    f"Failed to read sleep scoring file {self.sleep_scoring_file}"
                    f" with separator {separator}: {e}"
                )

        raise SleepScoringReadError(
            f"Failed to read sleep scoring file {self.sleep_scoring_file}"
            f" with default separators {_SLEEP_SCORING_FILE_SEPARATORS}"
        )


class ZMaxDataset(ABC):
    def __init__(
        self,
        data_dir: Path | str,
        zmax_dir_pattern: str,
        sleep_scoring_dir: Path | str | None = None,
        sleep_scoring_file_pattern: str | None = None,
        hypnogram_mapping: dict[int, str] = settings.DEFAULTS["hynogram_mapping"],
    ):
        self.data_dir = Path(data_dir)
        self._zmax_dir_pattern = zmax_dir_pattern
        self._sleep_scoring_dir = Path(sleep_scoring_dir) if sleep_scoring_dir else None
        self._sleep_scoring_file_pattern = sleep_scoring_file_pattern
        self.hypnogram_mapping = hypnogram_mapping

    def get_recordings(
        self, with_sleep_scoring: bool = False
    ) -> Generator[ZMaxRecording, None, None]:
        for zmax_dir in self._zmax_dir_generator():
            subject_id, session_id = self._extract_ids_from_zmax_dir(zmax_dir)
            recording = self._create_recording(subject_id, session_id, zmax_dir)

            if with_sleep_scoring and not recording.sleep_scoring_file:
                continue

            yield recording

    def _zmax_dir_generator(self) -> Generator[Path, None, None]:
        yield from self.data_dir.glob(self._zmax_dir_pattern)

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


def read_hypnogram(
    recording: ZMaxRecording,
    hypnogram_mapping: dict[int, str] | None = None,
    default_hypnogram_label: str = settings.DEFAULTS["hypnogram_label"],
) -> np.ndarray:
    stages = recording.read_sleep_scoring()[
        _SLEEP_SCORING_FILE_COLUMNS[0]
    ].values.squeeze()
    logger.debug(f"Stages shape: {stages.shape}")

    if hypnogram_mapping is not None:
        stages = mapper(hypnogram_mapping)(stages, default_hypnogram_label)

    return stages
