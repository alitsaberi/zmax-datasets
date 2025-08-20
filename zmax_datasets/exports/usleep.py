from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.datasets.base import (
    Dataset,
    Recording,
)
from zmax_datasets.exports.base import ExportStrategy
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.exports.utils import DataTypeMapping, SleepAnnotations
from zmax_datasets.transforms.resample import Resample
from zmax_datasets.utils.data import Data
from zmax_datasets.utils.exceptions import (
    ChannelDurationMismatchError,
    MissingDataTypeError,
    RawDataReadError,
    SleepScoringFileNotFoundError,
    SleepScoringFileNotSet,
    SleepScoringReadError,
)


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
        data_type_mappings: list[DataTypeMapping],
        sample_rate: float | None = None,
        annotation_type: SleepAnnotations | None = None,
        existing_file_handling: ExistingFileHandling = ExistingFileHandling.RAISE,
        data_type_error_handling: ErrorHandling = ErrorHandling.RAISE,
        annotation_error_handling: ErrorHandling = ErrorHandling.RAISE,
        error_handling: ErrorHandling = ErrorHandling.RAISE,
    ):
        super().__init__(
            existing_file_handling=existing_file_handling, error_handling=error_handling
        )
        self.data_type_mappings = data_type_mappings
        self.sample_rate = sample_rate
        self.annotation_type = annotation_type
        self.data_type_error_handling = data_type_error_handling
        self.annotation_error_handling = annotation_error_handling
        self._resample = Resample(sample_rate) if sample_rate is not None else None

    def _export(self, dataset: Dataset, out_dir: Path) -> None:
        prepared_recordings = 0
        catalog_data = []

        for i, recording in enumerate(dataset.get_recordings(with_sleep_scoring=False)):
            logger.info(f"-> Recording {i+1}: {recording}")

            recording_out_dir = out_dir / str(recording)
            recording_out_dir.mkdir(parents=True, exist_ok=True)

            # Initialize record info for catalog
            record_info = {
                "recording_id": str(recording),
                "has_annotations": False,
                # Will be updated to 'success' if export succeeds
                "export_status": "failed",
                **{
                    data_type_mapping.output_label: False
                    for data_type_mapping in self.data_type_mappings
                },
                **{f"{label}_count": 0 for label in dataset.hypnogram_mapping.values()},
            }

            try:
                # Extract and catalog data types
                data_info = self._extract_data_types(recording, recording_out_dir)
                record_info.update(data_info)

                # Extract and catalog annotations
                if self.annotation_type is not None:
                    annotation_info = self._extract_hypnogram(
                        recording, recording_out_dir, dataset.hypnogram_mapping
                    )
                    record_info.update(annotation_info)

                record_info["export_status"] = "success"
                prepared_recordings += 1
            except (
                MissingDataTypeError,
                RawDataReadError,
                SleepScoringReadError,
                SleepScoringFileNotFoundError,
                SleepScoringFileNotSet,
                ChannelDurationMismatchError,
                FileExistsError,
            ) as e:
                self._handle_error(e, recording)
                record_info["error_message"] = str(e)

            catalog_data.append(record_info)

        self._create_catalog(catalog_data, out_dir)

        logger.info(f"Prepared {prepared_recordings} recordings for USleep")

    def _extract_data_types(
        self,
        recording: Recording,
        recording_out_dir: Path,
    ) -> dict:
        logger.info("Extracting data types...")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['data_types_file_extension']}"
        )

        data_types_info = {}
        index = 0
        extracted_data_types = []
        expected_duration = None

        if not self._handle_existing_file(out_file_path):
            return data_types_info

        with h5py.File(out_file_path, "w") as out_file:
            out_file.create_group("channels")
            for data_type_mapping in self.data_type_mappings:
                data_type_name = data_type_mapping.output_label
                data_types_info[f"{data_type_name}"] = False

                if self._resample is not None:
                    data_type_mapping.transforms.append(self._resample)

                try:
                    data = data_type_mapping.map(recording)
                except (MissingDataTypeError, RawDataReadError) as e:
                    if self.data_type_error_handling == ErrorHandling.SKIP:
                        logger.warning(
                            f"Skipping {data_type_name} for recording {recording} "
                            f"because: {str(e)}"
                        )
                        continue
                    raise e

                # Validate length
                if expected_duration is None:
                    expected_duration = data.duration
                    data_types_info["length"] = str(data.length)
                    data_types_info["duration"] = str(expected_duration)

                elif data.duration != expected_duration:
                    raise ChannelDurationMismatchError(
                        f"Data type {data_type_name} has duration {data.duration}, "
                        f"expected {expected_duration}"
                    )

                self._write_data_to_hdf5(out_file, data_type_name, data, index)
                extracted_data_types.append(data_type_name)

                data_types_info[f"{data_type_name}"] = True

                index += 1

            logger.info(
                f"Extracted {len(extracted_data_types)} data types for recording "
                f"{recording}: {', '.join(extracted_data_types)}"
            )

            if self.sample_rate is not None:
                out_file.attrs["sample_rate"] = self.sample_rate

        return data_types_info

    def _write_data_to_hdf5(
        self, out_file: h5py.File, data_type_name: str, data: Data, index: int
    ) -> None:
        dataset = out_file["channels"].create_dataset(
            data_type_name,
            data=data.array.squeeze(),
            chunks=True,
            compression="gzip",
        )
        dataset.attrs["channel_index"] = index
        dataset.attrs["sample_rate"] = data.sample_rate

    def _extract_hypnogram(
        self,
        recording: Recording,
        recording_out_dir: Path,
        label_mapping: dict[int, str],
    ) -> dict:
        logger.info("Extracting hypnogram...")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['hypnogram_file_extension']}"
        )

        if not self._handle_existing_file(out_file_path):
            return {"has_annotations": True}

        annotation_info = {"has_annotations": False}

        try:
            annotations = recording.read_annotations(
                annotation_type=self.annotation_type, label_mapping=label_mapping
            )
            annotation_info["has_annotations"] = True

            # Add annotation stats
            unique_labels = np.unique(annotations)
            for label in unique_labels:
                label_count = np.sum(annotations == label)
                annotation_info[f"{label}_count"] = int(label_count)

            # Write to file
            initials, durations, annotations = ndarray_to_ids_format(annotations)
            initials, durations, annotations = squeeze_ids(
                initials, durations, annotations
            )

            with open(out_file_path, "w") as out_file:
                for i, d, s in zip(initials, durations, annotations, strict=False):
                    out_file.write(f"{int(i)},{int(d)},{s}\n")

            return annotation_info

        except (
            SleepScoringReadError,
            SleepScoringFileNotFoundError,
            SleepScoringFileNotSet,
        ) as e:
            if self.annotation_error_handling == ErrorHandling.SKIP:
                logger.warning(f"Skipping hypnogram for recording {recording}: {e}")
                return annotation_info
            raise e

    def _create_catalog(self, catalog_data: list[dict], out_dir: Path) -> None:
        catalog_path = out_dir / settings.USLEEP["catalog_file"]
        df = pd.DataFrame(catalog_data)
        df.to_csv(catalog_path, index=False)
        logger.info(f"Dataset catalog saved to {catalog_path}")

    def _handle_existing_file(self, file_path: Path) -> bool:
        """Handle existing file based on existing_file_handling setting.

        Returns:
            bool: True if processing should continue, False if it should be skipped
        """
        if file_path.exists():
            message = f"File {file_path} already exists."
            if self.existing_file_handling == ExistingFileHandling.SKIP:
                logger.info(f"{message} Skipping...")
                return False
            elif self.existing_file_handling == ExistingFileHandling.OVERWRITE:
                logger.info(f"{message} Overwriting...")
                file_path.unlink()
                return True
            else:
                raise FileExistsError(message)
        return True

    def _handle_error(
        self,
        error: Exception,
        recording: Recording,
    ) -> None:
        if self.error_handling == ErrorHandling.SKIP:
            logger.warning(f"Skipping recording {recording}: {error}")
        elif self.error_handling == ErrorHandling.RAISE:
            raise error
