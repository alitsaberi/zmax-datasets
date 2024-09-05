import logging
from pathlib import Path

import h5py
import numpy as np

from zmax_datasets import settings
from zmax_datasets.datasets.base import (
    ZMaxDataset,
    ZMaxRecording,
    read_hypnogram,
)
from zmax_datasets.datasets.utils import resample
from zmax_datasets.exports.base import ExportStrategy
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.utils.exceptions import MissingDataTypesError, SleepScoringReadError
from zmax_datasets.utils.helpers import remove_tree

logger = logging.getLogger(__name__)


def ndarray_to_ids_format(
    stages: np.ndarray, period_length: int = settings.DEFAULTS["period_length"]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_stages = len(stages)
    initials = np.arange(0, num_stages * period_length, period_length)
    durations = np.full(num_stages, period_length)
    return initials, durations, stages


def squeeze_ids(
    initials: np.ndarray, durations: np.ndarray, annotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    changes = np.concatenate(([True], (annotations[1:] != annotations[:-1])))
    squeezed_initials = initials[changes]
    squeezed_annotations = annotations[changes]
    squeezed_durations = np.diff(
        np.concatenate((squeezed_initials, [initials[-1] + durations[-1]]))
    )
    return squeezed_initials, squeezed_durations, squeezed_annotations


class USleepExportStrategy(ExportStrategy):
    def __init__(
        self,
        data_types: list[str],
        data_type_labels: dict[str, str] | None = None,
        sampling_frequency: int = settings.USLEEP["sampling_frequency"],
        existing_file_handling: ExistingFileHandling = ExistingFileHandling.RAISE_ERROR,
        error_handling: ErrorHandling = ErrorHandling.RAISE,
    ):
        super().__init__(
            existing_file_handling=existing_file_handling, error_handling=error_handling
        )
        self.data_types = data_types

        if data_type_labels and (
            invalid_data_types := set(data_type_labels.keys()) - set(data_types)
        ):
            logger.warning(f"Invalid keys in data_type_labels: {invalid_data_types}")

        self.data_type_labels = data_type_labels
        self.sampling_frequency = sampling_frequency

    def _export(self, dataset: ZMaxDataset, out_dir: Path) -> None:
        prepared_recordings = 0
        for i, recording in enumerate(dataset.get_recordings(with_sleep_scoring=True)):
            logger.info(f"-> Recording {i+1}: {recording}")

            recording_out_dir = out_dir / str(recording)
            recording_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                self._extract_data_types(recording, recording_out_dir)
                self._extract_hypnogram(
                    recording, recording_out_dir, dataset.hypnogram_mapping
                )
                prepared_recordings += 1
            except (
                MissingDataTypesError,
                SleepScoringReadError,
                FileExistsError,
                FileNotFoundError,
            ) as e:
                self._handle_error(e, recording, recording_out_dir)

        logger.info(f"Prepared {prepared_recordings} recordings for USleep")

    def _extract_data_types(
        self,
        recording: ZMaxRecording,
        recording_out_dir: Path,
    ) -> None:
        logger.info("Extracting data types...")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['data_types_file_extension']}"
        )
        self._check_existing_file(out_file_path)

        with h5py.File(out_file_path, "w") as out_file:
            out_file.create_group("channels")
            for index, data_type in enumerate(self.data_types):
                data = self._get_resampled_data(recording, data_type)
                self._write_data_to_hdf5(out_file, data_type, data, index)
            out_file.attrs["sample_rate"] = self.sampling_frequency

    def _get_resampled_data(
        self, recording: ZMaxRecording, data_type: str
    ) -> np.ndarray:
        raw = recording.read_raw_data(data_type)
        return resample(
            raw.get_data().squeeze(), self.sampling_frequency, raw.info["sfreq"]
        )

    def _write_data_to_hdf5(
        self, out_file: h5py.File, data_type: str, data: np.ndarray, index: int
    ) -> None:
        dataset = out_file["channels"].create_dataset(
            self.data_type_labels.get(data_type, data_type),
            data=data,
            chunks=True,
            compression="gzip",
        )
        dataset.attrs["channel_index"] = index

    def _extract_hypnogram(
        self,
        recording: ZMaxRecording,
        recording_out_dir: Path,
        hypnogram_mapping: dict[int, str],
    ) -> None:
        logger.info("Extracting hypnogram...")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['hypnogram_file_extension']}"
        )
        self._check_existing_file(out_file_path)

        stages = read_hypnogram(recording, hypnogram_mapping=hypnogram_mapping)

        initials, durations, stages = ndarray_to_ids_format(stages)
        initials, durations, stages = squeeze_ids(initials, durations, stages)

        with open(out_file_path, "w") as out_file:
            for i, d, s in zip(initials, durations, stages, strict=False):
                out_file.write(f"{int(i)},{int(d)},{s}\n")

    def _check_existing_file(self, file_path: Path) -> None:
        if (
            file_path.exists()
            and self.existing_file_handling == ExistingFileHandling.RAISE_ERROR
        ):
            raise FileExistsError(f"File {file_path} already exists.")

    def _handle_error(
        self, error: Exception, recording: ZMaxRecording, recording_out_dir: Path
    ) -> None:
        remove_tree(recording_out_dir)
        if self.error_handling == ErrorHandling.SKIP:
            logger.warning(f"Skipping recording {recording}: {error}")
        elif self.error_handling == ErrorHandling.RAISE:
            raise error
