import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from zmax_datasets import settings
from zmax_datasets.datasets.utils import mapper
from zmax_datasets.utils.exceptions import (
    MultipleSleepScoringFilesFoundError,
    SleepScoringFileNotFoundError,
    SleepScoringFileNotSet,
    SleepScoringReadError,
)
from zmax_datasets.utils.transforms import resample

logger = logging.getLogger(__name__)

EEG_CHANNELS: list[str] = ["EEG L", "EEG R"]
DATA_TYPES: list[str] = EEG_CHANNELS + [
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
_IGNORED_MODULES = ["__init__", "base", "utils"]


class SleepAnnotations(Enum):
    SLEEP_STAGE = "sleep_stage"
    AROUSAL = "arousal"


@dataclass
class ZMaxRecording:
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
    def data_types(self) -> list[str]:
        return [
            data_type
            for data_type_file in self.data_dir.glob(
                f"*.{settings.ZMAX['data_types_file_extension']}"
            )
            if (data_type := data_type_file.stem) in DATA_TYPES
        ]

    def __str__(self) -> str:
        return f"{self.subject_id}_{self.session_id}"

    def read_raw_data(self, data_type: str) -> np.ndarray:
        """Extract and return raw data from an EDF file for a specific data type.

        This method reads an EDF file corresponding to the given data type
        and returns the raw MNE object without further processing.

        Args:
            data_type (str): The type of data to extract (e.g., 'EEG L', 'EEG R').

        Returns:
            np.ndarray: The raw data for the specified data type.

        Raises:
            FileNotFoundError: If the EDF file for the specified data type is not found.
        """
        logger.info(f"Extracting {data_type}")
        data_type_file = (
            self.data_dir / f"{data_type}.{settings.ZMAX['data_types_file_extension']}"
        )
        raw = mne.io.read_raw_edf(data_type_file, preload=False)
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


@dataclass
class DataTypeMapping:
    output_label: str
    input_data_types: list[str]
    transforms: list[
        Callable[[np.ndarray], np.ndarray]
        | tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]
    ] = field(default_factory=list)

    def map(self, recording: ZMaxRecording, sampling_frequency: float) -> np.ndarray:
        data = self._get_raw_data(recording)
        logger.debug(f"Raw data shape: {data.shape}")
        data = self._transform_data(data, sampling_frequency)
        logger.debug(
            f"Processed data shape: {data.shape},"
            f" sampling frequency: {sampling_frequency}"
        )
        return data

    def _get_raw_data(self, recording: ZMaxRecording) -> np.ndarray:
        data_list = []

        for input_data_type in self.input_data_types:
            data = recording.read_raw_data(input_data_type)
            data_list.append(data)

        return np.vstack(data_list) if len(data_list) > 1 else data_list[0]

    def _transform_data(
        self, data: np.ndarray, sampling_frequency: float
    ) -> np.ndarray:
        for transform in self.transforms:
            if isinstance(transform, tuple):
                data = transform[0](data, **transform[1])
            else:
                data = transform(data)

        return resample(data, sampling_frequency, settings.ZMAX["sampling_frequency"])


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


def read_annotations(
    recording: ZMaxRecording,
    annotation_type: SleepAnnotations = SleepAnnotations.SLEEP_STAGE,
    label_mapping: dict[int, str] | None = None,
    default_label: str = settings.DEFAULTS["label"],
) -> np.ndarray:
    annotations = recording.read_sleep_scoring()[annotation_type.value].values.squeeze()
    logger.debug(f"{annotation_type.value} annotations shape: {annotations.shape}")

    if label_mapping is not None:
        annotations = mapper(label_mapping)(annotations, default_label)

    return annotations


def load_dataset_classes() -> dict[str, type[ZMaxDataset]]:
    datasets = {}
    datasets_dir = Path(__file__).parent

    for file_path in datasets_dir.rglob("*.py"):  # Recursively find all .py files
        module_name = file_path.stem

        if module_name in _IGNORED_MODULES:
            continue

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Inspect for classes that are subclasses of the base class
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, ZMaxDataset) and obj is not ZMaxDataset:
                datasets[name] = obj

    return datasets
