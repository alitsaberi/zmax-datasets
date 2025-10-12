from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.datasets.base import (
    Dataset as BaseDataset,
)
from zmax_datasets.datasets.base import (
    DataType,
    SleepAnnotations,
)
from zmax_datasets.datasets.base import (
    Recording as BaseRecording,
)
from zmax_datasets.sources.zmax.enums import DataTypes
from zmax_datasets.utils.exceptions import (
    MultipleSleepScoringFilesFoundError,
    SleepScoringFileNotFoundError,
    SleepScoringFileNotSet,
    SleepScoringReadError,
)

_SLEEP_SCORING_FILE_SEPARATORS = [" ", "\t"]


@dataclass
class Recording(BaseRecording):
    data_dir: Path
    subject_id: str | None = None
    session_id: str | None = None
    # TODO: change sleep_scoring to annotations in all files for consistent naming
    _sleep_scoring_file: Path | None = field(default=None, repr=False, init=False)

    @property
    def id(self) -> str:
        string_parts = []

        if self.subject_id is not None:
            string_parts.append(self.subject_id)
        if self.session_id is not None:
            string_parts.append(self.session_id)
        if not string_parts:
            string_parts.append(str(self.data_dir.name))
        return "-".join(string_parts)

    @property
    def sleep_scoring_file(self) -> Path | None:
        return self._sleep_scoring_file

    @sleep_scoring_file.setter
    def sleep_scoring_file(self, value: Path | None) -> None:
        if value is not None and not value.is_file():
            raise FileNotFoundError(f"Sleep scoring file {value} does not exist.")
        self._sleep_scoring_file = value

    @property
    def data_types(self) -> dict[str, DataType]:
        return {
            data_type.channel: data_type.value
            for file_path in self.data_dir.glob(
                f"*.{settings.ZMAX['data_types_file_extension']}"
            )
            if (data_type := DataTypes.get_by_channel(file_path.stem)) is not None
        }

    def _read_raw_data(self, data_type: DataType) -> np.ndarray:
        file_path = (
            self.data_dir
            / f"{data_type.channel}.{settings.ZMAX['data_types_file_extension']}"
        )

        raw = mne.io.read_raw_edf(file_path, preload=False)

        logger.debug(f"Channels: {raw.info['chs']}")

        return raw.get_data().squeeze()

    def _read_annotations(
        self,
        annotation_type: SleepAnnotations,
    ) -> np.ndarray:
        return self._read_sleep_scoring()[annotation_type.value].values.squeeze()

    def _read_sleep_scoring(self) -> pd.DataFrame:
        if self._sleep_scoring_file is None:
            raise SleepScoringFileNotSet(
                f"The sleep scoring file is not set for recording {self}"
            )

        for separator in _SLEEP_SCORING_FILE_SEPARATORS:
            try:
                return pd.read_csv(
                    self.sleep_scoring_file,
                    sep=separator,
                    names=[annotation.value for annotation in SleepAnnotations],
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


class Dataset(BaseDataset, ABC):
    def __init__(
        self,
        data_dir: Path | str,
        zmax_dir_pattern: str,
        sleep_scoring_dir: Path | str | None = None,
        sleep_scoring_file_pattern: str | None = None,
        hypnogram_mapping: dict[int, str] = settings.DEFAULTS["hypnogram_mapping"],
    ):
        self._zmax_dir_pattern = zmax_dir_pattern
        self._sleep_scoring_dir = Path(sleep_scoring_dir) if sleep_scoring_dir else None
        self._sleep_scoring_file_pattern = sleep_scoring_file_pattern
        super().__init__(data_dir, hypnogram_mapping)

    def get_recordings(
        self, with_sleep_scoring: bool = False
    ) -> Generator[Recording, None, None]:
        for zmax_dir in self._zmax_dir_generator():
            subject_id, session_id = self._extract_ids_from_zmax_dir(zmax_dir)
            recording = self._create_recording(subject_id, session_id, zmax_dir)

            if with_sleep_scoring and not recording.sleep_scoring_file:
                continue

            yield recording

    def _zmax_dir_generator(self) -> Generator[Path, None, None]:
        for zmax_dir in self.data_dir.glob(self._zmax_dir_pattern):
            if zmax_dir.is_dir():
                yield zmax_dir

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
    ) -> Recording:
        recording = Recording(
            data_dir=zmax_dir, subject_id=subject_id, session_id=session_id
        )
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
    def _get_sleep_scoring_file(self, recording: Recording) -> Path: ...
