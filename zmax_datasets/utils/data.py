from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger


class NoSamplesError(ValueError): ...


def _validate_channel_names_match(objects: Sequence["ArrayBase"]) -> list[str]:
    reference_channels = objects[0].channel_names
    for obj in objects[1:]:
        if obj.channel_names != reference_channels:
            raise ValueError(
                "All objects must have identical channel names."
                f" Reference: {reference_channels}, got: {obj.channel_names}"
            )
    return reference_channels


def _validate_channel_names_distinct(objects: Sequence["ArrayBase"]) -> list[str]:
    channel_names = []
    for obj in objects:
        if intersection := (set(channel_names) & set(obj.channel_names)):
            raise ValueError(
                "Channel names must be distinct."
                f" Duplicate channels found: {intersection}"
            )
        channel_names.extend(obj.channel_names)

    return channel_names


def _validate_timestamps_match(objects: Sequence["TimestampedArray"]) -> np.ndarray:
    reference_timestamps = objects[0].timestamps
    for obj in objects[1:]:
        if not np.array_equal(obj.timestamps, reference_timestamps):
            raise ValueError(
                "All objects must have identical timestamps."
                f" Reference: {reference_timestamps}, got: {obj.timestamps}"
            )
    return reference_timestamps


def _validate_sample_rate_match(objects: Sequence["Data"]) -> float:
    reference_sample_rate = objects[0].sample_rate
    for obj in objects[1:]:
        if obj.sample_rate != reference_sample_rate:
            raise ValueError(
                "All objects must have identical sample rates."
                f" Reference: {reference_sample_rate}, got: {obj.sample_rate}"
            )
    return reference_sample_rate




@dataclass
class ArrayBase:
    array: np.ndarray[Any, np.dtype[np.float64]]
    channel_names: list[str]

    def __post_init__(self):
        if not isinstance(self.array, np.ndarray):
            raise TypeError("array must be a numpy.ndarray")

        if self.channel_names is None:
            self.channel_names = [f"channel_{i}" for i in range(self.n_channels)]

        if len(self.channel_names) != self.n_channels:
            raise ValueError(
                f"Number of channel names ({len(self.channel_names)})"
                f" must match number of channels ({self.n_channels})."
            )

    # TODO: use dataclass methods for serialization
    def __str__(self) -> str:
        attrs = [f"{k}={v}" for k, v in self.attributes.items()]
        return f"{self.__class__.__name__}(shape={self.shape}, {', '.join(attrs)})"

    @property
    def attributes(self) -> dict[str, Any]:
        return {
            "channel_names": self.channel_names,
        }

    @property
    def datasets(self) -> dict[str, np.ndarray]:
        return {
            "data": self.array,
        }

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape of data"""
        return self.array.shape

    @property
    def n_channels(self) -> int:
        """Returns the number of channels in data"""
        return self.array.shape[-1]

    @cached_property
    def channel_index_map(self) -> dict[str, int]:
        """Returns a mapping from channel name to index"""
        return {name: idx for idx, name in enumerate(self.channel_names)}

    def to_csv(self, path: Path) -> None:
        """Saves data to a CSV file"""

        if path.exists():
            logger.warning(f"File {path} already exists, overwriting")

        with open(path, "w") as f:
            f.write(",".join(self.channel_names) + "\n")
            for row in self.array:
                f.write(",".join(map(str, row)) + "\n")

    @classmethod
    def concatenate(
        cls,
        objects: Sequence["ArrayBase"],
    ) -> "ArrayBase":
        """Concatenate multiple arrays along the time axis (vertically).

        All input arrays must have the same number and names of channels.

        Args:
            objects: Sequence of ArrayBase objects to concatenate

        Returns:
            A new ArrayBase object with arrays concatenated along time axis

        Raises:
            ValueError: If input arrays have different channels or
                less than two objects are provided.
        """
        if len(objects) < 2:
            raise ValueError("At least two objects must be provided.")

        # Validate channel names match
        channel_names = _validate_channel_names_match(objects)

        # Concatenate arrays along time axis
        array = np.concatenate([obj.array for obj in objects], axis=0)

        return cls(
            array=array,
            channel_names=channel_names,
        )

    @classmethod
    def stack_channels(cls, objects: Sequence["ArrayBase"]) -> "ArrayBase":
        """Stack multiple arrays by concatenating their channels horizontally.

        All input arrays must have the same length (number of samples).

        Args:
            objects: Sequence of ArrayBase objects to stack

        Returns:
            A new ArrayBase object with arrays stacked along channel axis

        Raises:
            ValueError: If input arrays have different lengths or no objects
                provided
        """
        if len(objects) < 2:
            raise ValueError("At least two objects must be provided.")

        channel_names = _validate_channel_names_distinct(objects)
        array = np.concatenate([obj.array for obj in objects], axis=1)

        return cls(
            array=array,
            channel_names=channel_names,
        )


@dataclass
class Sample(ArrayBase):
    timestamp: int  # nanoseconds since epoch

    def __post_init__(self):
        super().__post_init__()

        if self.array.ndim != 1:
            raise ValueError(
                f"Data must be 1D with shape (n_channels,), got shape {self.shape}"
            )

    @property
    def datasets(self) -> dict[str, np.ndarray]:
        return {
            "data": self.array.reshape(-1, 1),
            "timestamp": np.array([self.timestamp], dtype=np.int64),
        }


@dataclass
class TimestampedArray(ArrayBase):
    timestamps: np.ndarray[np.int64]  # nanoseconds since epoch

    @property
    def length(self) -> int:
        """Returns the number of samples in data"""
        return self.shape[0]

    @property
    def index(self) -> np.ndarray:
        """
        Return the time index of the data relative to the start time.
        """
        return self.timestamps - self.timestamps[0]

    @property
    def datasets(self) -> dict[str, np.ndarray]:
        return {
            "data": self.array,
            "timestamp": self.timestamps,
        }

    def __post_init__(self):
        super().__post_init__()

        if self.array.ndim != 2:
            raise ValueError(
                f"Data must be 2D with shape (n_samples, n_channels), "
                f"got shape {self.shape}"
            )

        if self.timestamps.shape != (self.length,):
            raise ValueError(
                f"Timestamps must have shape (n_samples,), "
                f"got shape {self.timestamps.shape}"
            )

    def __getitem__(
        self,
        key: slice
        | np.ndarray[np.bool_]
        | tuple[slice | np.ndarray[np.bool_], int | str | list[int] | list[str]],
    ) -> "TimestampedArray":
        """Support for array-like slicing with [channel_names/indices]"""
        if not isinstance(key, tuple):
            key = (key, None)

        samples, channels = key

        # Handle single index case to maintain 2D array
        if isinstance(samples, int):
            samples = slice(samples, samples + 1)

        if isinstance(samples, np.ndarray) and samples.dtype != bool:
            raise ValueError(f"Mask must be a boolean array, got {samples.dtype}")

        if isinstance(channels, str) or (
            isinstance(channels, list) and all(isinstance(c, str) for c in channels)
        ):
            return self.loc(samples, channels)

        return self.iloc(samples, channels)

    def __setitem__(
        self,
        key: slice | tuple[slice, int | str | list[int] | list[str]],
        value: "TimestampedArray",
    ) -> None:
        """Support for array-like assignment with [start:stop, channel_names/indices]"""

        if not isinstance(value, self.__class__):
            raise TypeError(f"Assignment value must be a {self.__class__} instance.")

        if not isinstance(key, tuple):
            self.array[key] = value.array
            self.timestamps[key] = value.timestamps
            return

        samples, channels = key

        self.timestamps[samples] = value.timestamps

        if isinstance(channels, str):
            channels = [channels]

        if isinstance(channels, list) and all(isinstance(c, str) for c in channels):
            channel_indices = self.get_channel_indices(channels)
            self.array[samples, channel_indices] = value
        else:
            self.array[samples, channels] = value

    def loc(
        self,
        samples: slice | np.ndarray[np.bool_] | None = None,
        channels: str | list[str] | None = None,
    ) -> "TimestampedArray":
        if channels is None:
            channels = self.channel_names

        if isinstance(channels, str):
            channels = [channels]

        channel_indices = self.get_channel_indices(channels)
        return self._slice_data(samples, channel_indices, channels)

    def iloc(
        self,
        samples: slice | np.ndarray[np.bool_] | None = None,
        channels: int | list[int] | None = None,
    ) -> "TimestampedArray":
        if channels is None:
            channels = list(range(self.n_channels))
        if isinstance(channels, int):
            channels = [channels]
        channel_names = [self.channel_names[i] for i in channels]
        return self._slice_data(samples, channels, channel_names)

    def get_channel_indices(self, channels: list[str]) -> list[int]:
        return [self.channel_index_map[ch] for ch in channels]

    def _slice_data(
        self,
        samples: slice | np.ndarray[np.bool_] | None,
        channels: list,
        channel_names: list[str],
    ) -> "TimestampedArray":
        samples = samples if samples is not None else slice(None)
        kwargs = self._get_slice_kwargs(samples, channels, channel_names)
        return type(self)(**kwargs)

    def _get_slice_kwargs(
        self,
        samples: slice | np.ndarray[np.bool_],
        channels: list[int],
        channel_names: list[str],
    ) -> dict[str, Any]:
        return {
            "array": self.array[samples][:, channels],
            "channel_names": channel_names,
            "timestamps": self.timestamps[samples],
        }

    def roll(self, shift: int) -> None:
        self.array = np.roll(self.array, shift, axis=0)
        self.timestamps = np.roll(self.timestamps, shift)

    @classmethod
    def concatenate(cls, objects: Sequence["TimestampedArray"]) -> "TimestampedArray":
        base = super().concatenate(objects)

        timestamps = np.concatenate([obj.timestamps for obj in objects])
        # TODO: check if timestamps are sorted

        return cls(
            array=base.array,
            channel_names=base.channel_names,
            timestamps=timestamps,
        )

    @classmethod
    def stack_channels(
        cls,
        objects: Sequence["TimestampedArray"],
    ) -> "TimestampedArray":
        base = super().stack_channels(objects)

        timestamps = _validate_timestamps_match(objects)

        return cls(
            array=base.array,
            channel_names=base.channel_names,
            timestamps=timestamps,
        )

    def slice_by_time(
        self, start_timestamp: int | None = None, end_timestamp: int | None = None
    ) -> "TimestampedArray":
        if start_timestamp is None:
            start_timestamp = self.timestamps[0]
        if end_timestamp is None:
            end_timestamp = self.timestamps[-1]

        mask = (self.timestamps >= start_timestamp) & (self.timestamps <= end_timestamp)
        return self[mask]


@dataclass
class Data(TimestampedArray):
    """
    Represents a time series of data with constant sample rate.
    """

    sample_rate: float

    def __init__(
        self,
        array: np.ndarray,
        sample_rate: float,
        channel_names: list[str] | None = None,
        timestamp_offset: int = 0,  # nanoseconds
        timestamps: np.ndarray | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")

        self.array = array
        self.sample_rate = sample_rate
        timestamps = timestamps if timestamps is not None else self.index
        timestamps = timestamps + timestamp_offset

        super().__init__(
            array=array,
            channel_names=channel_names,
            timestamps=timestamps,
        )

    def _get_slice_kwargs(
        self, samples: slice, channels: list, channel_names: list[str]
    ) -> dict[str, Any]:
        kwargs = super()._get_slice_kwargs(samples, channels, channel_names)
        kwargs["sample_rate"] = self.sample_rate
        return kwargs

    def __setitem__(
        self,
        key: slice | tuple[slice, int | str | list[int] | list[str]],
        value: "Data",
    ) -> None:
        if value.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {value.sample_rate} != {self.sample_rate}"
            )

        super().__setitem__(key, value)

    @property
    def index(self) -> np.ndarray:
        return (
            np.arange(self.length, dtype=np.int64) * (1e9 / self.sample_rate)
        ).astype(np.int64)

    @property
    def duration(self) -> timedelta:
        return timedelta(microseconds=self.length * (1e9 / self.sample_rate) / 1000)

    @property
    def attributes(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            **super().attributes,
        }

    @classmethod
    def concatenate(
        cls,
        objects: Sequence["Data"],
    ) -> "Data":
        base = super().concatenate(objects)

        sample_rate = _validate_sample_rate_match(objects)

        return Data(
            array=base.array,
            channel_names=base.channel_names,
            timestamps=base.timestamps,
            sample_rate=sample_rate,
        )

    @classmethod
    def stack_channels(
        cls,
        objects: Sequence["Data"],
    ) -> "Data":
        if len(objects) < 2:
            raise ValueError("At least two objects must be provided.")

        channel_names = _validate_channel_names_distinct(objects)
        timestamps = _validate_timestamps_match(objects)
        sample_rate = _validate_sample_rate_match(objects)
        array = np.concatenate([obj.array for obj in objects], axis=1)

        return Data(
            array=array,
            channel_names=channel_names,
            timestamps=timestamps,
            sample_rate=sample_rate,
        )


@dataclass
class DataType:
    channel: str
    sampling_rate: float

    @property
    def label(self) -> str:
        return "_".join(self.channel.split(" "))


def samples_to_timestamped_array(samples: Sequence[Sample]) -> TimestampedArray:
    if len(samples) == 0:
        raise NoSamplesError("No samples provided")

    channels_names = _validate_channel_names_match(samples)

    array = np.stack([sample.array for sample in samples])
    timestamps = np.array([sample.timestamp for sample in samples], dtype=np.int64)

    return TimestampedArray(
        array=array,
        channel_names=channels_names,
        timestamps=timestamps,
    )


def get_all_periods_by_period_length(
    data: Data,
    period_length: int,
) -> np.ndarray:
    """
    Returns all periods in data.
    Args:
        data (Data): The data to get periods from.
        period_length (int): The length of each period in seconds.
    Returns:
        np.ndarray: An numpy array.
                    Shape (n_periods, n_samples_per_period, n_channels)
    """
    n_samples_per_period = int(period_length * data.sample_rate)
    return get_all_periods(data, n_samples_per_period)


def get_all_periods(data: Data, n_samples_per_period: int) -> np.ndarray:
    """
    Returns all periods in data.
    Args:
        data (Data): The data to get periods from.
        n_samples_per_period (int): The number of samples in each period.
    Returns:
        np.ndarray: An numpy array.
                    Shape (n_periods, n_samples_per_period, n_channels)
    """
    return get_periods_by_index(data, 0, n_samples_per_period, None)


def get_periods_by_index(
    data: Data,
    start_index: int,
    n_samples_per_period: int,
    n_periods: int | None = None,
) -> np.ndarray:
    """
    Returns a number of periods in data starting from a given index.

    Args:
        data (Data): The data to get periods from.
        start_index (int): The index of the first period to return.
        n_samples_per_period (int): The number of samples in each period.
        n_periods (int): The number of periods to return. If None,
                            all periods from the start index to the end of
                            the data are returned.

    Returns:
        np.ndarray: An numpy array
                    Shape: (n_periods, n_samples_per_period, n_channels)

    Raises:
        ValueError: If the requested number of periods exceeds
                    the length of the data.
    """
    if start_index < 0 or start_index >= data.length // n_samples_per_period:
        raise ValueError(f"Invalid start_index: {start_index}")

    if n_samples_per_period <= 0 or n_samples_per_period > data.length:
        raise ValueError(f"Invalid n_samples_per_period: {n_samples_per_period}")

    start_sample_index = start_index * n_samples_per_period
    n_available_samples = data.length - start_sample_index

    if n_periods is None:
        if data.length % n_samples_per_period != 0:
            raise ValueError(
                f"Data length {data.length} is not"
                f" a multiple of {n_samples_per_period}."
            )
        n_periods = n_available_samples // n_samples_per_period

    end_sample_index = start_sample_index + (n_samples_per_period * n_periods)

    if end_sample_index > data.length:
        raise ValueError(
            f"Requested {n_periods} periods, but only"
            f" {n_available_samples // n_samples_per_period} periods are available."
        )

    period_indices = np.arange(start_sample_index, end_sample_index)
    period_indices = period_indices.reshape(n_periods, n_samples_per_period)

    return data.array[period_indices]

def epochify(
    data: Data,
    *,
    epoch_duration: float,
    expected_sample_rate: float | None = None,
    drop_remainder: bool = True,
) -> tuple[np.ndarray, int, int, Data]:
    """Convert a continuous Data object into fixed-length epochs.

    Args:
        data: Input continuous data with shape (n_samples, n_channels).
        epoch_duration: Epoch duration in seconds.
        expected_sample_rate: If provided, validate that
            ``data.sample_rate`` matches this value.
        drop_remainder: If True, drop samples at the end that do not make a
            complete epoch. If False, raise an error when the data length is
            not an exact multiple of the epoch length.

    Returns:
        A tuple containing:
            - array: NumPy array of shape (n_epochs, n_samples_per_epoch, n_channels)
            - n_epochs: Number of epochs.
            - epoch_length: Number of samples per epoch.
            - trimmed_data: Input data trimmed so that it contains only
              full epochs.

    Raises:
        ValueError: If ``epoch_duration`` is not positive, if
            ``expected_sample_rate`` does not match ``data.sample_rate``,
            or if ``drop_remainder`` is False and the data length is not
            a multiple of the epoch length.
        NoSamplesError: If no complete epochs can be formed from the data.
    """
    
    # TODO: revisit whether sample-rate validation should be handled
    #       at the pipeline level rather than in generic utilities.
    if expected_sample_rate is not None and data.sample_rate != expected_sample_rate:
        raise ValueError(
            f"Data must have sample rate {expected_sample_rate}, got {data.sample_rate}"
        )

    if epoch_duration <= 0:
        raise ValueError(f"epoch_duration must be positive, got {epoch_duration}")

    epoch_length = int(epoch_duration * data.sample_rate)
    if epoch_length <= 0:
        raise ValueError(
            f"epoch_duration={epoch_duration} too small for sample_rate={data.sample_rate}"
        )

    n_epochs = data.length // epoch_length
    if n_epochs == 0:
        raise NoSamplesError(
            f"No complete epochs found: data_length={data.length}, epoch_length={epoch_length}"
        )

    samples_to_keep = n_epochs * epoch_length

    logger.info(
        f"[epochify] samples={data.length}, epoch_length={epoch_length}, "
        f"n_epochs={n_epochs}, samples_to_keep={samples_to_keep}"
    )
    
    # TODO: support padding incomplete epochs (e.g., zero- or NaN-padding)
    if samples_to_keep < data.length:
        if not drop_remainder:
            raise ValueError(
                f"Data length {data.length} is not a multiple of epoch_length {epoch_length}."
            )
        logger.info(f"[epochify] Dropping {data.length - samples_to_keep} samples at the end.")
        data = data[:samples_to_keep]

    array = data.array.reshape(
        n_epochs, epoch_length, data.n_channels
    )  # (N, T, C)

    return array, n_epochs, epoch_length, data



def detect_voltage_unit(
    values: np.ndarray,
    *,
    microvolt_threshold: float = 1e-3,
) -> Literal["volts", "microvolts"]:
    """Heuristically detect whether a signal is in volts or microvolts.

    Detection is based on the absolute magnitude of the signal.
    This is a heuristic and should not be considered authoritative.

    Args:
        values: Signal values.
        microvolt_threshold: Threshold above which values are assumed
            to be in microvolts. Defaults to 1e-3.

    Returns:
        "volts" if the signal appears to be in volts,
        "microvolts" if the signal appears to be in microvolts.
    """
    max_abs = np.nanmax(np.abs(values))

    if max_abs > microvolt_threshold:
        return "microvolts"

    return "volts"



def microvolts_to_volts(values: np.ndarray) -> np.ndarray:
    """Convert microvolts (µV) to volts (V).

    Args:
        values: Array of values in microvolts.

    Returns:
        Array of values in volts.
    """
    return values * 1e-6


def volts_to_microvolts(values: np.ndarray) -> np.ndarray:
    """Convert volts (V) to microvolts (µV).

    Args:
        values: Array of values in volts.

    Returns:
        Array of values in microvolts.
    """
    return values * 1e6

def ensure_volts(values: np.ndarray) -> np.ndarray:
    """Ensure signal is in volts, converting from microvolts if needed."""
    unit = detect_voltage_unit(values)
    return microvolts_to_volts(values) if unit == "microvolts" else values

def ensure_microvolts(values: np.ndarray) -> np.ndarray:
    """Ensure signal is in microvolts, converting from volts if needed."""
    unit = detect_voltage_unit(values)
    return volts_to_microvolts(values) if unit == "volts" else values

