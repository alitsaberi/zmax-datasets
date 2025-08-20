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
from zmax_datasets.utils.exceptions import MissingDataTypeError, RawDataReadError


class Recording(ABC):
    @cached_property
    def data_types(self) -> dict[str, DataType]:
        raise NotImplementedError

    def read_raw_data(
        self,
        data_type_label: str,
    ) -> tuple[np.ndarray, float]:
        logger.info(f"Extracting {data_type_label}")

        if data_type_label not in self.data_types:
            raise MissingDataTypeError(
                f"Data type {data_type_label} not found in {self}"
            )

        data_type = self.data_types[data_type_label]
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

    @abstractmethod
    def _read_raw_data(self, data_type: DataType) -> np.ndarray: ...

    @abstractmethod
    def read_annotations(
        self,
        annotation_type: SleepAnnotations,
        label_mapping: dict[int, str] | None = None,
        default_label: str = settings.DEFAULTS["label"],
    ) -> np.ndarray: ...


class Dataset(ABC):
    def __init__(
        self,
        data_dir: Path | str,
        hypnogram_mapping: dict[int, str] = settings.DEFAULTS["hynogram_mapping"],
    ):
        self.data_dir = Path(data_dir)
        self.hypnogram_mapping = hypnogram_mapping

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
