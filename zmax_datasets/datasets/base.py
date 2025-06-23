from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any

import numpy as np
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.datasets.utils import mapper
from zmax_datasets.utils.exceptions import MissingDataTypeError
from zmax_datasets.utils.transforms import resample


class SleepAnnotations(Enum):
    SLEEP_STAGE = "sleep_stage"
    AROUSAL = "arousal"


@dataclass
class DataType:
    channel: str
    sampling_rate: float

    @property
    def label(self) -> str:
        return self.channel.split(" ").join("_")


class Recording(ABC):
    @cached_property
    def data_types(self) -> dict[str, DataType]:
        raise NotImplementedError

    def read_raw_data(
        self, data_type_label: str, sampling_frequency: float | None = None
    ) -> np.ndarray:
        logger.info(f"Extracting {data_type_label}")
        if data_type_label not in self.data_types:
            raise MissingDataTypeError(
                f"Data type {data_type_label} not found in {self}"
            )

        data_type = self.data_types[data_type_label]
        data = self._read_raw_data(data_type)

        if sampling_frequency is not None:
            logger.info(
                f"Resampling {data_type_label} from"
                f" {data_type.sampling_rate} Hz"
                f" to {sampling_frequency} Hz"
            )
            data = resample(data, sampling_frequency, data_type.sampling_rate)

        return data

    @abstractmethod
    def _read_raw_data(self, data_type: DataType) -> np.ndarray: ...

    @abstractmethod
    def read_annotations(
        self,
        annotation_type: SleepAnnotations,
        label_mapping: dict[int, str] | None = None,
        default_label: str = settings.DEFAULTS["label"],
    ) -> np.ndarray: ...


@dataclass
class DataTypeMapping:
    output_label: str
    input_data_types: list[str]
    transforms: list[
        Callable[[np.ndarray], np.ndarray]
        | tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]
    ] = field(default_factory=list)

    def map(self, recording: Recording, sampling_frequency: float) -> np.ndarray:
        data = self._get_raw_data(recording, sampling_frequency)
        logger.debug(f"Raw data shape: {data.shape}")
        data = self._transform_data(data)
        logger.debug(
            f"Processed data shape: {data.shape},"
            f" sampling frequency: {sampling_frequency}"
        )
        return data

    def _get_raw_data(
        self, recording: Recording, sampling_frequency: float
    ) -> np.ndarray:
        data_list = []

        for data_type_label in self.input_data_types:
            data = recording.read_raw_data(data_type_label, sampling_frequency)
            data_list.append(data)

        return np.vstack(data_list) if len(data_list) > 1 else data_list[0]

    def _transform_data(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        for transform in self.transforms:
            if isinstance(transform, tuple):
                data = transform[0](data, **transform[1])
            else:
                data = transform(data)

        return data


class Dataset(ABC):
    @abstractmethod
    def get_recordings(
        self, with_sleep_scoring: bool = False
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
