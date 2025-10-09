from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.pipeline.engine import execute_pipeline

if TYPE_CHECKING:
    from zmax_datasets.pipeline.configs import PipelineConfig

from zmax_datasets.datasets.base import (
    Dataset,
    Recording,
)
from zmax_datasets.exports.base import ExportStrategy
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.exports.utils import SleepAnnotations
from zmax_datasets.transforms.helpers import Resample
from zmax_datasets.utils.data import Data
from zmax_datasets.utils.exceptions import (
    ChannelDurationMismatchError,
    MissingDataTypeError,
    RawDataReadError,
    SampleRateNotFoundError,
    SleepScoringFileNotFoundError,
    SleepScoringFileNotSet,
    SleepScoringReadError,
)

CATALOG_FILE_NAME = "catalog"
CATALOG_FILE_EXTENSION = "csv"
ANNOTATIONS_DATA_TYPE = "annotations"


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


def _handle_existing_file(
    file_path: Path, existing_file_handling: ExistingFileHandling
) -> bool:
    """Handle existing file based on existing_file_handling setting.

    Returns:
        bool: True if processing should continue, False if it should be skipped
    """
    if file_path.exists():
        message = f"File {file_path} already exists."
        if existing_file_handling == ExistingFileHandling.SKIP:
            logger.info(f"{message} Skipping...")
            return False
        elif existing_file_handling == ExistingFileHandling.OVERWRITE:
            logger.info(f"{message} Overwriting...")
            file_path.unlink()
            return True
        elif existing_file_handling == ExistingFileHandling.APPEND:
            logger.info(f"{message} Appending...")
            return True
        else:
            raise FileExistsError(message)
    return True


class USleepExportStrategy(ExportStrategy):
    def __init__(
        self,
        input_data_types: list[str] | None = None,
        output_data_types: list[str] | None = None,
        rename_mapping: dict[str, str] | None = None,
        pipeline_config: "PipelineConfig | None" = None,
        sample_rate: float | None = None,
        annotation_type: SleepAnnotations | None = None,
        existing_file_handling: ExistingFileHandling = ExistingFileHandling.RAISE,
        data_type_error_handling: ErrorHandling = ErrorHandling.RAISE,
        annotation_error_handling: ErrorHandling = ErrorHandling.RAISE,
        error_handling: ErrorHandling = ErrorHandling.RAISE,
        with_sleep_scoring: bool = False,
        catalog_file_name: str = CATALOG_FILE_NAME,
        period_length: int = settings.DEFAULTS["period_length"],
    ):
        super().__init__(
            existing_file_handling=existing_file_handling, error_handling=error_handling
        )

        self.input_data_types = input_data_types
        self.output_data_types = output_data_types
        self.rename_mapping = rename_mapping or {}
        self.pipeline_config = pipeline_config
        self.sample_rate = sample_rate
        self.annotation_type = annotation_type
        self.catalog_file_name = f"{catalog_file_name}.{CATALOG_FILE_EXTENSION}"
        self.existing_data_file_handling = self.existing_file_handling
        self.existing_annotation_file_handling = (
            self.existing_file_handling
            if self.existing_file_handling != ExistingFileHandling.APPEND
            else ExistingFileHandling.OVERWRITE
        )
        self.data_type_error_handling = data_type_error_handling
        self.annotation_error_handling = annotation_error_handling
        self._resample = Resample(sample_rate) if sample_rate is not None else None
        self.with_sleep_scoring = with_sleep_scoring
        self.include_annotations_in_pipeline = False

        if pipeline_config:
            for step in pipeline_config.steps:
                if ANNOTATIONS_DATA_TYPE in step.input_data_types:
                    self.include_annotations_in_pipeline = True
                    break

        self.period_length = period_length

    def _get_processed_recordings(self, out_dir: Path) -> set[str]:
        """Get set of recording IDs that were successfully processed."""
        catalog_path = out_dir / self.catalog_file_name
        if not catalog_path.exists():
            return set()

        df = pd.read_csv(catalog_path)
        processed = df["recording_id"].tolist()
        logger.info(f"Found {len(processed)} processed recordings in catalog")
        return set(processed)

    def _export(self, dataset: Dataset, out_dir: Path) -> None:
        # Read successful recordings once at the start
        processed_recordings = self._get_processed_recordings(out_dir)
        n_new_processed_recordings = 0

        for i, recording in enumerate(
            dataset.get_recordings(with_sleep_scoring=self.with_sleep_scoring)
        ):
            logger.info(f"-> Recording {i+1}: {recording}")
            recording_id = str(recording)

            # Skip if already successfully processed
            if recording_id in processed_recordings:
                logger.info(f"Skipping recording {recording_id} - already processed")
                continue

            # Set up recording directory
            out_dir_path = out_dir / recording_id
            out_dir_path.mkdir(parents=True, exist_ok=True)

            # Initialize record info for catalog
            record_info = {
                "recording_id": recording_id,
                "has_annotations": False,
                "export_status": "failed",
                **{data_type: False for data_type in self.output_data_types},
            }

            # Only add label counts if annotation_type is set
            if self.annotation_type is not None:
                # Initialize all possible label counts to 0
                for label in settings.DEFAULTS["hypnogram_mapping"].values():
                    record_info[f"{label}_count"] = 0

            try:
                # 1. Load data
                annotations = self._load_annotations(
                    recording, dataset.hypnogram_mapping
                )
                data_types = self._load_data(recording, annotations)

                # 2. Process data
                data_types = self._process_data(data_types)

                # 3. Export processed data
                data_info = self._write_data_types(data_types, out_dir_path, recording)
                record_info.update(data_info)

                # 4. Export annotations
                if ANNOTATIONS_DATA_TYPE in data_types:
                    annotations = data_types.pop(ANNOTATIONS_DATA_TYPE)

                annotation_info = self._write_hypnogram(
                    annotations, out_dir_path, recording, dataset.hypnogram_mapping
                )
                record_info.update(annotation_info)

                record_info["export_status"] = "success"
                n_new_processed_recordings += 1

            except (
                MissingDataTypeError,
                RawDataReadError,
                SleepScoringReadError,
                SleepScoringFileNotFoundError,
                SleepScoringFileNotSet,
                ChannelDurationMismatchError,
                SampleRateNotFoundError,
                FileExistsError,
                FileNotFoundError,
                ValueError,
            ) as e:
                self._handle_error(e, recording)
                record_info["error_message"] = str(e)

            # Update catalog after each recording
            self._update_catalog(record_info, out_dir)

        logger.info(f"Prepared {n_new_processed_recordings} recordings.")

    def _load_annotations(
        self,
        recording: Recording,
        label_mapping: dict[int, str],
    ) -> Data | None:
        """Load annotations from recording.

        Args:
            recording: Recording to load annotations from
            label_mapping: Mapping from numeric labels to string labels

        Returns:
            Annotations Data object or None if not available
        """
        if self.annotation_type is not None:
            try:
                return recording.read_annotations(
                    annotation_type=self.annotation_type,
                    label_mapping=label_mapping,
                )
            except (
                SleepScoringReadError,
                SleepScoringFileNotFoundError,
                SleepScoringFileNotSet,
            ) as e:
                if self.annotation_error_handling == ErrorHandling.SKIP:
                    logger.warning(f"Skipping hypnogram for recording {recording}: {e}")
                else:
                    raise e

        return None

    def _load_data(
        self,
        recording: Recording,
        annotations: Data | None,
    ) -> dict[str, Data]:
        """Load data types and prepare for processing.

        Args:
            recording: Recording to load data from
            annotations: Annotations array if available

        Returns:
            Dictionary of loaded data types
        """
        # Load data types
        data_types = {}
        for label in self.input_data_types:
            try:
                data_types[label] = recording.read_data_type(label)
            except (MissingDataTypeError, RawDataReadError) as e:
                if self.data_type_error_handling == ErrorHandling.SKIP:
                    logger.warning(
                        f"Skipping data type {label} for recording {recording}: {e}"
                    )
                else:
                    raise e

        logger.info(f"Read initial data types: {list(data_types.keys())}")

        # Apply renaming if specified
        if self.rename_mapping:
            renamed_data_types = {}
            for original_name, data in data_types.items():
                new_name = self.rename_mapping.get(original_name, original_name)
                renamed_data_types[new_name] = data
                if new_name != original_name:
                    logger.info(f"Renamed data type: {original_name} -> {new_name}")
            data_types = renamed_data_types

        # Add annotations if needed for pipeline
        if self.include_annotations_in_pipeline and annotations is not None:
            data_types[ANNOTATIONS_DATA_TYPE] = annotations
            logger.info("Added annotations data to data types")

        return data_types

    def _process_data(
        self,
        data_types: dict[str, Data],
    ) -> dict[str, Data]:
        """Process loaded data types through pipeline and resampling.

        Args:
            data_types: Dictionary of data types to process

        Returns:
            Dictionary of processed data types
        """
        # Execute pipeline if provided
        if self.pipeline_config:
            data_types = execute_pipeline(
                pipeline_config=self.pipeline_config,
                initial_data=data_types,
                output_data_types=None,  # Get all pipeline outputs
            )
            logger.info(f"Pipeline produced data types: {list(data_types.keys())}")

        # Apply resampling if specified (skip sleep_stage)
        if self._resample:
            for data_type, data in data_types.items():
                if data_type != ANNOTATIONS_DATA_TYPE:
                    data_types[data_type] = self._resample(data)

        return data_types

    def _write_data_types(
        self,
        data_types: dict[str, Data],
        recording_out_dir: Path,
        recording: Recording,
    ) -> dict:
        """Write data types to HDF5 file."""
        logger.info("Writing data types...")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['data_types_file_extension']}"
        )

        if not _handle_existing_file(out_file_path, self.existing_data_file_handling):
            return {}

        mode = (
            "w"
            if self.existing_data_file_handling == ExistingFileHandling.OVERWRITE
            else "a"
        )

        if self.sample_rate is not None and out_file_path.exists():
            with h5py.File(out_file_path, "r") as f:
                if (file_sample_rate := f.attrs.get("sample_rate")) != self.sample_rate:
                    raise ValueError(
                        f"Sample rate mismatch:"
                        f" file has {file_sample_rate} Hz,"
                        f" but {self.sample_rate} Hz was provided"
                    )

        # Write outputs to h5 file
        data_types_info = {}
        extracted_data_types = []
        expected_duration = None

        with h5py.File(out_file_path, mode) as out_file:
            # Create channels group if it doesn't exist
            if "channels" not in out_file:
                out_file.create_group("channels")

            # Get next channel index for appending
            index = len(out_file["channels"].keys())

            for data_type in self.output_data_types:
                data_types_info[data_type] = False

                if data_type not in data_types:
                    if self.data_type_error_handling == ErrorHandling.SKIP:
                        logger.warning(f"Skipping missing data type: {data_type}")
                        continue
                    else:
                        raise MissingDataTypeError(
                            f"Data type {data_type} not found in output"
                        )

                data = data_types[data_type]

                # Validate length consistency
                if expected_duration is None:
                    expected_duration = data.duration
                    data_types_info["length"] = str(data.length)
                    data_types_info["duration"] = str(expected_duration)
                elif data.duration != expected_duration:
                    raise ChannelDurationMismatchError(
                        f"Data type {data_type} has duration {data.duration}, "
                        f"expected {expected_duration}"
                    )

                # If appending, check if channel already exists
                if data_type in out_file["channels"]:
                    logger.info(f"Overwriting existing channel: {data_type}")
                    del out_file["channels"][data_type]

                # Calculate and store channel statistics as separate columns
                channel_stats = self._calculate_channel_stats(data)
                for stat_name, stat_value in channel_stats.items():
                    data_types_info[f"{data_type}_{stat_name}"] = stat_value

                self._write_data_to_hdf5(out_file, data_type, data, index)
                extracted_data_types.append(data_type)
                data_types_info[data_type] = True
                index += 1

            logger.info(
                f"Wrote {len(extracted_data_types)} data types: "
                f"{', '.join(extracted_data_types)}"
            )

            if self.sample_rate is not None:
                out_file.attrs["sample_rate"] = self.sample_rate

        return data_types_info

    def _write_hypnogram(
        self,
        annotations: Data | None,
        recording_out_dir: Path,
        recording: Recording,
        label_mapping: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Write hypnogram to file in IDS format and calculate stats.

        Args:
            annotations: Annotations Data object or None
            recording_out_dir: Directory to write hypnogram to
            recording: Recording being processed
            label_mapping: Optional mapping from numeric to string labels

        Returns:
            Dictionary with annotation stats
        """
        annotation_info = {"has_annotations": False}

        if annotations is None:
            return annotation_info

        logger.info("Writing hypnogram...")
        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.USLEEP['hypnogram_file_extension']}"
        )

        if not _handle_existing_file(
            out_file_path, self.existing_annotation_file_handling
        ):
            return annotation_info

        # Calculate annotation stats
        annotation_info["has_annotations"] = True
        annotations_array = annotations.array.squeeze()
        # Use default mapping if none provided
        mapping = label_mapping or settings.DEFAULTS["hypnogram_mapping"]
        for label in mapping.values():
            label_count = np.sum(annotations_array == label)
            annotation_info[f"{label}_count"] = int(label_count)

        # Convert to IDS format and write
        initials, durations, stages = ndarray_to_ids_format(
            annotations_array, self.period_length
        )
        initials, durations, stages = squeeze_ids(initials, durations, stages)

        with open(out_file_path, "w") as out_file:
            for i, d, s in zip(initials, durations, stages, strict=False):
                out_file.write(f"{int(i)},{int(d)},{s}\n")

        logger.info(f"Wrote hypnogram to {out_file_path}")
        return annotation_info

    def _calculate_channel_stats(self, data: Data) -> dict:
        """Calculate basic statistics for a channel."""
        array = data.array.squeeze()
        return {
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
        }

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

    def _update_catalog(self, record_info: dict, out_dir: Path) -> None:
        """Update catalog with new record info."""
        catalog_path = out_dir / self.catalog_file_name

        # Read existing catalog if it exists
        if catalog_path.exists():
            df = pd.read_csv(catalog_path)
            # Update or append record
            idx = df[df["recording_id"] == record_info["recording_id"]].index
            if len(idx) > 0:
                df.loc[idx[0]] = pd.Series(record_info)
            else:
                df = pd.concat([df, pd.DataFrame([record_info])], ignore_index=True)
        else:
            df = pd.DataFrame([record_info])

        df.to_csv(catalog_path, index=False)
        logger.info(f"Updated catalog at {catalog_path}")

    def _handle_error(
        self,
        error: Exception,
        recording: Recording,
    ) -> None:
        if self.error_handling == ErrorHandling.SKIP:
            logger.warning(f"Skipping recording {recording}: {error}")
        elif self.error_handling == ErrorHandling.RAISE:
            raise error
