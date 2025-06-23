from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
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
from zmax_datasets.datasets.utils import mapper
from zmax_datasets.utils.exceptions import (
    MultipleSleepScoringFilesFoundError,
    SleepScoringFileNotFoundError,
    SleepScoringFileNotSet,
    SleepScoringReadError,
)

_SLEEP_SCORING_FILE_SEPARATORS = [" ", "\t"]


class DataTypes(Enum):
    EEG_RIGHT = DataType("EEG R", settings.ZMAX["sampling_frequency"])
    EEG_LEFT = DataType("EEG L", settings.ZMAX["sampling_frequency"])
    ACCELEROMETER_X = DataType("dX", settings.ZMAX["sampling_frequency"])
    ACCELEROMETER_Y = DataType("dY", settings.ZMAX["sampling_frequency"])
    ACCELEROMETER_Z = DataType("dZ", settings.ZMAX["sampling_frequency"])
    BODY_TEMP = DataType("BODY TEMP", settings.ZMAX["sampling_frequency"])
    BATTERY = DataType("BATT", settings.ZMAX["sampling_frequency"])
    NOISE = DataType("NOISE", settings.ZMAX["sampling_frequency"])
    LIGHT = DataType("LIGHT", settings.ZMAX["sampling_frequency"])
    NASAL_LEFT = DataType("NASAL L", settings.ZMAX["sampling_frequency"])
    NASAL_RIGHT = DataType("NASAL R", settings.ZMAX["sampling_frequency"])
    OXIMETER_INFRARED_AC = DataType("OXY_IR_AC", settings.ZMAX["sampling_frequency"])
    OXIMETER_RED_AC = DataType("OXY_R_AC", settings.ZMAX["sampling_frequency"])
    OXIMETER_DARK_AC = DataType("OXY_DARK_AC", settings.ZMAX["sampling_frequency"])
    OXIMETER_INFRARED_DC = DataType("OXY_IR_DC", settings.ZMAX["sampling_frequency"])
    OXIMETER_RED_DC = DataType("OXY_R_DC", settings.ZMAX["sampling_frequency"])
    OXIMETER_DARK_DC = DataType("OXY_DARK_DC", settings.ZMAX["sampling_frequency"])

    @property
    def category(self) -> str:
        return self.name.split("_")[0]

    @property
    def channel(self) -> str:
        return self.value.channel

    @classmethod
    def get_by_channel(cls, channel: str) -> DataType | None:
        for data_type in cls:
            if data_type.channel == channel:
                return data_type
        return None

    @classmethod
    def get_by_category(cls, category: str) -> list["DataType"]:
        return [data_type for data_type in cls if data_type.category == category]


@dataclass
class Recording(BaseRecording):
    subject_id: str
    session_id: str
    data_dir: Path
    # TODO: change sleep_scoring to annotations in all files for consistent naming
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
    def data_types(self) -> dict[str, DataType]:
        return {
            data_type.channel: data_type.value
            for file_path in self.data_dir.glob(
                f"*.{settings.ZMAX['data_types_file_extension']}"
            )
            if (data_type := DataTypes.get_by_channel(file_path.stem)) is not None
        }

    def __str__(self) -> str:
        return f"{self.subject_id}_{self.session_id}"

    def _read_raw_data(self, data_type: DataType) -> np.ndarray:
        file_path = (
            self.data_dir
            / f"{data_type.channel}.{settings.ZMAX['data_types_file_extension']}"
        )

        raw = mne.io.read_raw_edf(file_path, preload=False)
        logger.debug(f"Channels: {raw.info['chs']}")

        return raw.get_data().squeeze()

    def read_sleep_scoring(self) -> pd.DataFrame:
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

    def read_annotations(
        self,
        annotation_type: SleepAnnotations = SleepAnnotations.SLEEP_STAGE,
        label_mapping: dict[int, str] | None = None,
        default_label: str = settings.DEFAULTS["label"],
    ) -> np.ndarray:
        annotations = self.read_sleep_scoring()[annotation_type.value].values.squeeze()
        logger.debug(f"{annotation_type.value} annotations shape: {annotations.shape}")

        if label_mapping is not None:
            annotations = mapper(label_mapping)(annotations, default_label)

        return annotations


class Dataset(BaseDataset, ABC):
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
    ) -> Generator[Recording, None, None]:
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
    ) -> Recording:
        recording = Recording(subject_id, session_id, zmax_dir)
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
