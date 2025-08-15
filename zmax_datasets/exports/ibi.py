from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.datasets.base import (
    Dataset,
    DataTypeMapping,
    Recording,
    SleepAnnotations,
)
from zmax_datasets.exports.base import ExportStrategy
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.processing.ibi import extract_ibi
from zmax_datasets.utils.exceptions import (
    MissingDataTypeError,
    SleepScoringReadError,
)
from zmax_datasets.utils.helpers import remove_tree


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


class IBIExportStrategy(ExportStrategy):
    def __init__(
        self,
        data_type_mappings: list[DataTypeMapping],
        sampling_frequency: float = settings.IBI["sampling_frequency"],
        segment_duration: int = settings.IBI["segment_duration"],
        annotation_type: SleepAnnotations = SleepAnnotations.SLEEP_STAGE,
        existing_file_handling: ExistingFileHandling = ExistingFileHandling.RAISE_ERROR,
        missing_data_type_handling: ErrorHandling = ErrorHandling.RAISE,
        error_handling: ErrorHandling = ErrorHandling.RAISE,
    ):
        super().__init__(
            existing_file_handling=existing_file_handling, error_handling=error_handling
        )
        self.missing_data_type_handling = missing_data_type_handling
        self.data_type_mappings = data_type_mappings
        self.sampling_frequency = sampling_frequency
        self.segment_duration = segment_duration
        self.annotation_type = annotation_type

    def _export(self, dataset: Dataset, out_dir: Path) -> None:
        prepared_recordings = 0
        for i, recording in enumerate(dataset.get_recordings(with_sleep_scoring=True)):
            logger.info(f"-> Recording {i+1}: {recording}")

            recording_out_dir = out_dir / str(recording)
            recording_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                self._extract_ibi(recording, recording_out_dir)
                self._extract_hypnogram(
                    recording, recording_out_dir, dataset.hypnogram_mapping
                )
                prepared_recordings += 1
            except (
                MissingDataTypeError,
                SleepScoringReadError,
                FileExistsError,
                FileNotFoundError,
            ) as e:
                self._handle_error(e, recording, recording_out_dir)

        logger.info(f"Prepared {prepared_recordings} recordings for USleep")

    def _extract_ibi(
        self,
        recording: Recording,
        recording_out_dir: Path,
    ) -> None:
        logger.info("Extracting data types...")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['data_types_file_extension']}"
        )
        self._check_existing_file(out_file_path)

        index = 0
        with h5py.File(out_file_path, "w") as out_file:
            out_file.create_group("channels")
            for data_type_mapping in self.data_type_mappings:
                try:
                    data = data_type_mapping.map(recording)
                    logger.info(f"Extracting IBI from {data.shape} data")
                    ibi_signal, usability_labels = extract_ibi(
                        data,
                        128,
                        self.segment_duration,
                        self.sampling_frequency,
                    )
                    self._write_data_to_hdf5(out_file, "ibi", ibi_signal, index)
                    self._write_usability_labels(
                        usability_labels, recording, recording_out_dir
                    )
                    self._plot_and_save_signal(
                        ibi_signal,
                        recording_out_dir,
                        recording,
                        usability_labels,
                        self.segment_duration,
                    )
                except MissingDataTypeError as e:
                    if self.missing_data_type_handling == ErrorHandling.SKIP:
                        logger.warning(
                            f"Skipping data type {data_type_mapping.output_label} for"
                            f" recording {recording} because {e}"
                        )
                        continue

                    raise e

                index += 1

            if index == 0:
                raise MissingDataTypeError(
                    f"No data types found for recording {recording}"
                )

            out_file.attrs["sample_rate"] = self.sampling_frequency

    def _write_data_to_hdf5(
        self, out_file: h5py.File, data_type_name: str, data: np.ndarray, index: int
    ) -> None:
        dataset = out_file["channels"].create_dataset(
            data_type_name,
            data=data,
            chunks=True,
            compression="gzip",
        )
        dataset.attrs["channel_index"] = index

    def _write_usability_labels(
        self,
        usability_labels: np.ndarray,
        recording: Recording,
        recording_out_dir: Path,
    ) -> np.ndarray:
        logger.info("Saving usability labels...")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.ARTIFACT_DETECTION['labels_file_extension']}"
        )
        self._check_existing_file(out_file_path)

        unique, counts = np.unique(usability_labels, return_counts=True)
        label_counts = dict(zip(unique, counts, strict=False))
        logger.info(f"Unique label counts: {label_counts}")

        # Write usability scores to file
        with open(out_file_path, "w") as out_file:
            out_file.write("ibi\n")
            for label in usability_labels:
                out_file.write(f"{label}\n")

        logger.info(f"Saved usability labels to {out_file_path}")

    def _extract_hypnogram(
        self,
        recording: Recording,
        recording_out_dir: Path,
        label_mapping: dict[int, str],
    ) -> None:
        logger.info("Extracting hypnogram...")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['hypnogram_file_extension']}"
        )
        self._check_existing_file(out_file_path)

        annotations = recording.read_annotations(
            annotation_type=self.annotation_type, label_mapping=label_mapping
        )

        initials, durations, annotations = ndarray_to_ids_format(annotations)
        initials, durations, annotations = squeeze_ids(initials, durations, annotations)

        with open(out_file_path, "w") as out_file:
            for i, d, s in zip(initials, durations, annotations, strict=False):
                out_file.write(f"{int(i)},{int(d)},{s}\n")

    def _plot_and_save_signal(
        self,
        ibi_signal: np.ndarray,
        recording_out_dir: Path,
        recording: Recording,
        usability_labels: np.ndarray,
        segment_duration: int,
    ) -> None:
        plt.figure(figsize=(12, 4))
        time_in_hours = np.arange(len(ibi_signal)) / (
            self.sampling_frequency * 3600
        )  # Convert time to hours
        plt.plot(time_in_hours, ibi_signal, label="IBI Signal")
        plt.title(f"IBI Signal for {recording}")
        plt.xlabel("Time (hours)")
        plt.ylabel("IBI Value")
        plt.grid(True)

        # Highlight unusable segments with a grey transparent window
        num_segments = len(usability_labels)
        segment_times = np.arange(num_segments) * (
            segment_duration / 3600
        )  # Convert segment times to hours
        for i, label in enumerate(usability_labels):
            if label:  # If the segment is marked as unusable
                plt.axvspan(
                    segment_times[i],
                    segment_times[i] + (segment_duration / 3600),
                    color="grey",
                    alpha=0.5,
                    label="Unusable Segment" if i == 0 else "",
                )

        plt.legend()
        plt.tight_layout()
        plot_file_path = recording_out_dir / f"{recording}_ibi_signal.png"
        plt.savefig(plot_file_path)
        plt.close()
        logger.info(f"Saved IBI signal plot to {plot_file_path}")

    def _check_existing_file(self, file_path: Path) -> None:
        if (
            file_path.exists()
            and self.existing_file_handling == ExistingFileHandling.RAISE_ERROR
        ):
            raise FileExistsError(f"File {file_path} already exists.")

    def _handle_error(
        self, error: Exception, recording: Recording, recording_out_dir: Path
    ) -> None:
        remove_tree(recording_out_dir)
        if self.error_handling == ErrorHandling.SKIP:
            logger.warning(f"Skipping recording {recording}: {error}")
        elif self.error_handling == ErrorHandling.RAISE:
            raise error
