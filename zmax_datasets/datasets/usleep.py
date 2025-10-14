from collections.abc import Generator
from pathlib import Path

import h5py
import numpy as np

from zmax_datasets import settings
from zmax_datasets.datasets.base import Dataset, Recording
from zmax_datasets.exports.utils import SleepAnnotations
from zmax_datasets.utils.data import DataType
from zmax_datasets.utils.exceptions import (
    SampleRateNotFoundError,
    SleepScoringFileNotFoundError,
)

CHANNELS_GROUP = "channels"


class USleepRecording(Recording):
    def __init__(
        self,
        data_dir: Path,
    ):
        self._data_dir = data_dir

        self._data_file = (
            self._data_dir / f"{self.id}.{settings.USLEEP['data_types_file_extension']}"
        )
        self._annotation_file = (
            self._data_dir / f"{self.id}.{settings.USLEEP['hypnogram_file_extension']}"
        )

    @property
    def id(self) -> str:
        return self._data_dir.stem

    @property
    def data_types(self) -> dict[str, DataType]:
        """Get available data types from the HDF5 file."""
        data_types = {}
        with h5py.File(self._data_file, "r", locking=False) as f:
            for channel_name in f[CHANNELS_GROUP]:
                channel_data = f[CHANNELS_GROUP][channel_name]
                sample_rate = channel_data.attrs.get(
                    "sample_rate", f.attrs.get("sample_rate")
                )

                if sample_rate is None:
                    raise SampleRateNotFoundError(
                        f"Sample rate for channel {channel_name} can not be determined."
                    )

                data_types[channel_name] = DataType(
                    channel=channel_name,
                    sampling_rate=sample_rate,
                )

        return data_types

    def _read_raw_data(self, data_type: DataType) -> np.ndarray:
        """Read raw data from HDF5 file."""
        with h5py.File(self._data_file, "r", locking=False) as f:
            return f["channels"][data_type.channel][:]

    def _read_annotations(
        self,
        annotation_type: SleepAnnotations,
    ) -> np.ndarray:
        """Read sleep stage annotations from .ids file."""
        if self._annotation_file is None or not self._annotation_file.exists():
            raise SleepScoringFileNotFoundError(
                f"Annotation file not found for recording {self}"
            )

        if annotation_type == SleepAnnotations.AROUSAL:
            raise ValueError("Arousal annotations are not supported by USleep.")

        # Read the .ids file format: initial,duration,stage
        initials, durations, stages = [], [], []
        with open(self._annotation_file) as f:
            for line in f:
                i, d, s = line.strip().split(",")
                initials.append(int(i))
                durations.append(int(d))
                stages.append(s)

        # Convert to numpy arrays
        initials = np.array(initials)
        durations = np.array(durations)
        stages = np.array(stages)

        # Expand to full length array
        total_length = int(
            (initials[-1] + durations[-1]) / settings.DEFAULTS["period_length"]
        )
        full_stages = np.full(total_length, settings.DEFAULTS["label"], dtype=object)

        # Fill in the stages
        for i, d, s in zip(initials, durations, stages, strict=False):
            start_idx = i // settings.DEFAULTS["period_length"]
            end_idx = (i + d) // settings.DEFAULTS["period_length"]
            full_stages[start_idx:end_idx] = s

        return full_stages


class USleepDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
    ):
        super().__init__(data_dir)

    def get_recordings(
        self,
        with_sleep_scoring: bool = False,
    ) -> Generator[Recording, None, None]:
        """Get all recordings in the dataset."""
        for item in self.data_dir.iterdir():
            # Only process directories, skip files like catalog.csv
            if not item.is_dir():
                continue

            data_dir = item
            if with_sleep_scoring:
                annotation_file = (
                    data_dir
                    / f"{data_dir.stem}.{settings.USLEEP['hypnogram_file_extension']}"
                )
                if not annotation_file.exists():
                    continue

            yield USleepRecording(
                data_dir=data_dir,
            )
