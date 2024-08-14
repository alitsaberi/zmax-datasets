from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import h5py
import mne

from zmax_datasets import settings
from zmax_datasets.datasets.exceptions import MissingDataTypesError
from zmax_datasets.datasets.utils import resample

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
    subject_id: int
    session_id: int
    data_dir: Path
    sleep_scores_file: Path | None = None

    @property
    def data_types(self) -> list[str]:
        return [file.stem for file in self.data_dir.glob(f"*.{_DATA_TYPE_FILE_FORMAT}")]
    
    def __str__(self) -> str:
        return f"s{self.subject_id}_n{self.session_id}"


class ZMaxDataset(ABC):

    def __init__(
        self,
        data_dir: Path | str,
        selected_subjects: list[int] | None = None,
        selected_sessions: list[int] | None = None,
        excluded_subjects: list[int] | None = None,
        excluded_sessions: list[int] | None = None,
        excluded_sessions_for_subjects: dict[int, list[int]] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.selected_subjects = selected_subjects
        self.selected_sessions = selected_sessions
        self.excluded_subjects = excluded_subjects
        self.excluded_sessions = excluded_sessions
        self.excluded_sessions_for_subjects = excluded_sessions_for_subjects
        self.recordings_ = (
            self._collect_recordings()
        )  # TODO: Make this a cached property and return as a generator

    def __len__(self) -> int:
        return len(self._recordings)

    @abstractmethod
    def _collect_recordings(self) -> list[ZMaxRecording]: ...

    def _is_recording_included(self, subject_id: int, session_id: int) -> bool:
        return (
            (not self.selected_subjects or subject_id in self.selected_subjects)
            and (not self.excluded_subjects or subject_id not in self.excluded_subjects)
            and (not self.selected_sessions or session_id in self.selected_sessions)
            and (not self.excluded_sessions or session_id not in self.excluded_sessions)
            and (
                not self.excluded_sessions_for_subjects
                or subject_id not in self.excluded_sessions_for_subjects
                or session_id not in self.excluded_sessions_for_subjects[subject_id]
            )
        )

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

        total_recordings = len(self)
        prepared_recordings = 0
        logger.info(f"-> Preparing {total_recordings} recordings for USleep")
        for i, recording in enumerate(self._recordings):
            logger.info(
                f"-> Recording {i+1}/{total_recordings}: {recording}"
            )

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

            self._extract_data_types(
                recording,
                recording_out_dir,
                data_types,
                data_type_labels,
                sampling_frequency,
                existing_file_handling,
            )
            # self._extract_hypnogram(recording)

            prepared_recordings += 1
        logger.info(f"-> Prepared {prepared_recordings} recordings for USleep")

    @classmethod
    def _extract_data_types(
        cls,
        recording: ZMaxRecording,
        recording_out_dir: Path,
        data_types: list[str],
        data_type_labels: dict[str, str] | None = None,
        sampling_frequency: int = settings.USLEEP["sampling_frequency"],
        existing_file_handling: HandlingStrategy = HandlingStrategy.RAISE_ERROR,
    ) -> None:
        logger.info(f"-> Extracting data types: {data_types}")

        out_file_path = recording_out_dir / f"{recording_out_dir.name}.h5"

        if out_file_path.exists():
            if existing_file_handling == HandlingStrategy.RAISE_ERROR:
                raise FileExistsError(f"File {out_file_path} already exists.")
            elif existing_file_handling == HandlingStrategy.SKIP:
                logger.info(f"Skipping {out_file_path} because it already exists")
                return

        with h5py.File(out_file_path, "w") as out_file:
            out_file.create_group("channels")
            for index, data_type in enumerate(data_types):
                logger.info(f"-> Extracting {data_type}")
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
