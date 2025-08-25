import numpy as np
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.datasets.zmax import Recording
from zmax_datasets.sources.zmax.enums import DataTypes
from zmax_datasets.utils.data import Data


def _read_data_type(recording: Recording, data_type: DataTypes) -> np.ndarray:
    data, _ = recording.read_data_type(data_type.channel)
    if data_type.category == "EEG":
        data = data * 1_000_000  # Convert to microvolts
    return data


def load_data(recording: Recording, data_types: list[DataTypes]) -> Data:
    array = np.column_stack(
        [_read_data_type(recording, data_type) for data_type in data_types]
    )
    data = Data(
        array,
        sample_rate=settings.ZMAX["sampling_frequency"],
        channel_names=[data_type.name for data_type in data_types],
    )
    logger.debug(f"Loaded data: {data}")
    return data
