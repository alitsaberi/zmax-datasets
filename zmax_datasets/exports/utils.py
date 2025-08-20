from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data

if TYPE_CHECKING:
    from zmax_datasets.datasets.base import Recording


# TODO: rename to SleepAnnotation and check if this is the best place for it
class SleepAnnotations(Enum):
    SLEEP_STAGE = "sleep_stage"
    AROUSAL = "arousal"


@dataclass
class DataTypeMapping:
    output_label: str
    input_data_types: list[str]
    transforms: list[Transform] = field(default_factory=list)

    def map(self, recording: "Recording") -> Data:
        data = self._get_raw_data(recording)
        logger.debug(f"Raw data shape: {data.shape}")
        data = self._transform_data(data)
        logger.debug(f"Processed data shape: {data.shape}")
        return data

    def _get_raw_data(
        self,
        recording: "Recording",
    ) -> Data:
        data_list = []

        for data_type_label in self.input_data_types:
            data = recording.read_raw_data(data_type_label)
            data_list.append(data)

        return Data.stack_channels(data_list) if len(data_list) > 1 else data_list[0]

    def _transform_data(
        self,
        data: Data,
    ) -> Data:
        for transform in self.transforms:
            data = transform(data)

        return data
