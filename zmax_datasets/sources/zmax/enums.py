from enum import Enum
from typing import Optional

from zmax_datasets import settings
from zmax_datasets.utils.data import DataType


class DataTypes(Enum):  # TODO: rename to DataType and make DataType class Channel
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
    def get_by_channel(cls, channel: str) -> Optional["DataTypes"]:
        for data_type in cls:
            if data_type.channel == channel:
                return data_type
        return None

    @classmethod
    def get_by_category(cls, category: str) -> list["DataTypes"]:
        return [data_type for data_type in cls if data_type.category == category]
