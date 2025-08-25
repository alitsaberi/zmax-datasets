from abc import ABC, abstractmethod
from collections.abc import Generator
from functools import cached_property
from pathlib import Path

import numpy as np
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.datasets.utils import mapper
from zmax_datasets.exports.utils import SleepAnnotations
from zmax_datasets.utils.data import Data, DataType
from zmax_datasets.utils.exceptions import (
    MissingDataTypeError,
    RawDataReadError,
    RecordingNotFoundError,
)


class Recording(ABC):
    def __str__(self) -> str:
        return self.id

    @property
    @abstractmethod
    def id(self) -> str: ...

    @cached_property
    @abstractmethod
    def data_types(self) -> dict[str, DataType]: ...

    def read_data_type(
        self,
        data_type_label: str,
    ) -> Data:
        logger.info(f"Reading data type: {data_type_label}")

        data_type = self.data_types.get(data_type_label)

        if data_type is None:
            raise MissingDataTypeError(
                f"Data type {data_type_label} not found in {self}"
            )

        try:
            array = self._read_raw_data(data_type)
        except Exception as e:
            raise RawDataReadError(
                f"Failed to read raw data from {data_type}: {e}"
            ) from e
        return Data(
            array=array.reshape(-1, 1),
            sample_rate=data_type.sampling_rate,
            channel_names=[data_type.channel],
        )

    def read_data_types(self, data_type_labels: list[str]) -> dict[str, Data]:
        """
        Read multiple data types and return as dictionary.

        Args:
            data_type_labels: List of data type labels to read

        Returns:
            Dictionary mapping data type labels to Data objects

        Raises:
            MissingDataTypeError: If any requested data type is not available
            RawDataReadError: If reading any data type fails
        """
        logger.info(f"Reading data types: {data_type_labels}")

        result = {}
        for label in data_type_labels:
            result[label] = self.read_data_type(label)

        return result

    @abstractmethod
    def _read_raw_data(self, data_type: DataType) -> np.ndarray: ...

    def read_annotations(
        self,
        annotation_type: SleepAnnotations,
        label_mapping: dict[int, str] | None = None,
        default_label: str = settings.DEFAULTS["label"],
    ) -> np.ndarray:
        annotations = self._read_annotations(annotation_type, default_label)
        if label_mapping is not None:
            annotations = mapper(label_mapping)(annotations, default_label)
        return annotations

    @abstractmethod
    def _read_annotations(self, annotation_type: SleepAnnotations) -> np.ndarray: ...


class Dataset(ABC):
    def __init__(
        self,
        data_dir: Path | str,
        hypnogram_mapping: dict[int, str] | None = None,
        load_recordings: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.hypnogram_mapping = hypnogram_mapping
        self._recordings: dict[str, Recording] | None = None
        if load_recordings:
            self.load()

    @property
    def n_recordings(self) -> int:
        self._check_loaded()
        return len(self._recordings)

    @property
    def recording_ids(self) -> list[str]:
        self._check_loaded()
        return list(self._recordings.keys())

    def get_recording(self, id: str) -> Recording:
        self._check_loaded()

        recording = self._recordings.get(id)
        if recording is None:
            raise RecordingNotFoundError(f"Recording with id {id} not found.")

        return recording

    def _check_loaded(self) -> None:
        if self._recordings is None:
            raise RuntimeError("Dataset not loaded. Run `load()` first.")

    def load(self, with_sleep_scoring: bool = False) -> None:
        self._recordings = {
            str(recording): recording
            for recording in self.get_recordings(with_sleep_scoring)
        }

    # TODO: this method should be used for loading the recordings into the dataset
    # probably it should be an internal method and renamed
    # another method should be implemented and used to iterate over the recordings
    # e.g. __iter__
    @abstractmethod
    def get_recordings(
        self,
        with_sleep_scoring: bool = False,  # TODO: rename to with_annotations
    ) -> Generator[Recording, None, None]: ...


def read_annotations(
    recording: Recording,
    annotation_type: SleepAnnotations = SleepAnnotations.SLEEP_STAGE,
    label_mapping: dict[int, str] | None = None,
    default_label: str = settings.DEFAULTS["label"],
) -> np.ndarray:
    annotations = recording.read_sleep_scoring()[annotation_type.value].values.squeeze()
    logger.debug(f"{annotation_type.value} annotations shape: {annotations.shape}")

    if label_mapping is not None:
        annotations = mapper(label_mapping)(annotations, default_label)

    return annotations
