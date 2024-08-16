import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

from zmax_datasets import settings
from zmax_datasets.datasets.exceptions import (
    MissingDataTypesError,
    SleepScoringReadError,
)
from zmax_datasets.datasets.utils import (
    mapper,
    ndarray_to_ids_format,
    resample,
    squeeze_ids,
)

logger = logging.getLogger(__name__)

_DATA_TYPE_FILE_FORMAT: str = "edf"
_EEG_CHANNELS: list[str] = ["EEG L", "EEG R"]
_DATA_TYPES: list[str] = _EEG_CHANNELS + [
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


class HandlingStrategy(Enum):
    RAISE_ERROR = "raise_error"
    SKIP = "skip"
    OVERWRITE = "overwrite"


@dataclass
class ZMaxRecording:
    subject_id: str
    session_id: str
    data_dir: Path
    sleep_scores_file: Path | None = None

    @property
    def data_types(self) -> list[str]:
        return [file.stem for file in self.data_dir.glob(f"*.{_DATA_TYPE_FILE_FORMAT}")]

    def __str__(self) -> str:
        return f"s{self.subject_id}_n{self.session_id}"


class ZMaxDataset(ABC):
    _MANUAL_SLEEP_SCORES_FILE_SEPARATORS: list[str] = [" ", "\t"]

    def __init__(
        self,
        data_dir: Path | str,
    ):
        self.data_dir = Path(data_dir)

    @abstractmethod
    def get_recordings(self) -> Generator[ZMaxRecording, None, None]: ...

    def _process_zmax_dir(self, zmax_dir: Path) -> ZMaxRecording | None:
        subject_id, session_id = self._extract_ids_from_zmax_dir(
            zmax_dir
        )  # TODO: Raise error rather than removing None

        if subject_id is None or session_id is None:
            logger.debug(
                "Skipping recording with because"
                f"Could not extract subject and session IDs from {zmax_dir}"
            )
            return

        return ZMaxRecording(
            subject_id=subject_id,
            session_id=session_id,
            data_dir=zmax_dir,
            sleep_scores_file=self._get_sleep_scores_file(
                zmax_dir, subject_id, session_id
            ),
        )

    @classmethod
    @abstractmethod
    def _extract_ids_from_zmax_dir(
        cls, zmax_dir: Path
    ) -> tuple[int | None, int | None]: ...

    @abstractmethod
    def _get_sleep_scores_file(
        self, zmax_dir: Path, subject_id: int, session_id: int
    ) -> Path | None: ...

    def to_usleep(
        self,
        out_dir: Path,
        data_types: list[str],
        data_type_labels: dict[str, str] | None = None,
        sampling_frequency: int = settings.USLEEP["sampling_frequency"],
        existing_file_handling: HandlingStrategy = HandlingStrategy.RAISE_ERROR,
        missing_data_type_handling: HandlingStrategy = HandlingStrategy.RAISE_ERROR,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        if data_type_labels and (
            invalid_data_types := set(data_type_labels.keys()) - set(data_types)
        ):
            logger.warning(f"Invalid keys in data_type_labels: {invalid_data_types}")

        prepared_recordings = 0
        logger.info("Preparing recordings for USleep")
        for i, recording in enumerate(self.get_recordings()):
            logger.info(f"-> Recording {i+1}: {recording}")

            if recording.sleep_scores_file is None:
                logger.warning(
                    f"Skipping recording {recording} "
                    "because sleep scores file is missing"
                )
                continue

            recording_out_dir = out_dir / str(recording)
            recording_out_dir.mkdir(parents=True, exist_ok=True)

            logging.debug(
                f"Extracting data types: {data_types}."
                f" Available data types: {recording.data_types}"
            )
            if missing_data_types := set(data_types) - set(recording.data_types):
                if missing_data_type_handling == HandlingStrategy.RAISE_ERROR:
                    raise MissingDataTypesError(missing_data_types)
                elif missing_data_type_handling == HandlingStrategy.SKIP:
                    logger.warning(
                        f"Skipping recording {recording_out_dir.name}"
                        f" due to missing data types: {missing_data_types}"
                    )
                    continue

            self.extract_data_types(
                recording,
                recording_out_dir,
                data_types,
                data_type_labels,
                sampling_frequency,
                existing_file_handling,
            )

            try:
                self.extract_hypnogram(
                    recording, recording_out_dir, existing_file_handling
                )
            except SleepScoringReadError:
                logger.warning(
                    f"Skipping recording {recording} due to sleep scoring read error."
                )
                continue

            prepared_recordings += 1
        logger.info(f"Prepared {prepared_recordings} recordings for USleep")

    @classmethod
    def extract_data_types(
        cls,
        recording: ZMaxRecording,
        recording_out_dir: Path,
        data_types: list[str],
        data_type_labels: dict[str, str] | None = None,
        sampling_frequency: int = settings.USLEEP["sampling_frequency"],
        existing_file_handling: HandlingStrategy = HandlingStrategy.RAISE_ERROR,
    ) -> None:
        logger.info(f"-> Extracting data types: {data_types}")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['data_types_file_extension']}"
        )

        if out_file_path.exists():
            if existing_file_handling == HandlingStrategy.RAISE_ERROR:
                raise FileExistsError(f"File {out_file_path} already exists.")
            elif existing_file_handling == HandlingStrategy.SKIP:
                logger.info(f"Skipping {out_file_path} because it already exists")
                return

        with h5py.File(out_file_path, "w") as out_file:
            out_file.create_group("channels")
            for index, data_type in enumerate(data_types):
                logger.info(f"Extracting {data_type}")
                data_type_file = (
                    recording.data_dir / f"{data_type}.{_DATA_TYPE_FILE_FORMAT}"
                )
                raw = mne.io.read_raw_edf(
                    data_type_file, preload=False
                )  # TODO: Check whether it's required to convert the untis
                logger.debug(f"Channels: {raw.info['chs']}")
                data = raw.get_data().squeeze()

                data = resample(data, sampling_frequency, raw.info["sfreq"])

                dataset = out_file["channels"].create_dataset(
                    data_type_labels.get(data_type, data_type),
                    data=data,
                    chunks=True,
                    compression="gzip",
                )
                dataset.attrs["channel_index"] = index

            out_file.attrs["sample_rate"] = sampling_frequency

    @classmethod
    def extract_hypnogram(
        cls,
        recording: ZMaxRecording,
        recording_out_dir: Path,
        existing_file_handling: HandlingStrategy = HandlingStrategy.RAISE_ERROR,
    ) -> None:
        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['hypnogram_file_extension']}"
        )
        if out_file_path.exists():
            if existing_file_handling == HandlingStrategy.RAISE_ERROR:
                raise FileExistsError(f"File {out_file_path} already exists.")
            elif existing_file_handling == HandlingStrategy.SKIP:
                logger.info(f"Skipping {out_file_path} because it already exists")
                return

        stages = cls._read_hypnogram(recording.sleep_scores_file)
        logger.debug(f"Stages shape: {stages.shape}")

        stages = mapper(cls._USLEEP_HYPNOGRAM_MAPPING)(
            stages, settings.USLEEP["default_hypnogram_label"]
        )

        initials, durations, stages = ndarray_to_ids_format(
            stages,
            period_length=30,
        )
        initials, durations, stages = squeeze_ids(initials, durations, stages)

        with open(out_file_path, "w") as out_file:
            for i, d, s in zip(initials, durations, stages, strict=False):
                out_file.write(f"{int(i)},{int(d)},{s}\n")

    @classmethod
    def _read_hypnogram(cls, hypnogram_file: Path) -> np.ndarray:
        return cls._read_manual_sleep_scores(hypnogram_file)[
            "sleep_stage"
        ].values.squeeze()

    @classmethod
    def _read_manual_sleep_scores(cls, sleep_scores_file: Path) -> pd.DataFrame:
        for separator in cls._MANUAL_SLEEP_SCORES_FILE_SEPARATORS:
            try:
                return pd.read_csv(
                    sleep_scores_file,
                    sep=separator,
                    names=["sleep_stage", "arousal"],
                    dtype=int,
                )
            except ValueError:
                logger.debug(
                    f"Failed to read sleep scores file {sleep_scores_file}"
                    f" with separator {separator}. Trying next separator"
                )

        raise SleepScoringReadError(
            f"Failed to read hypnogram file {sleep_scores_file} with default separators"
            f" {cls._MANUAL_SLEEP_SCORES_FILE_SEPARATORS}"
        )
