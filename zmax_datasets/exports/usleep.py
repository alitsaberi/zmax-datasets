from pathlib import Path
from typing import TYPE_CHECKING

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
from zmax_datasets.transforms.resample import Resample
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
        input_data_types: list[str] | None = None,
        output_data_types: list[str] | None = None,
        pipeline_config: "PipelineConfig | None" = None,
        sample_rate: float | None = None,
        annotation_type: SleepAnnotations | None = None,
        existing_file_handling: ExistingFileHandling = ExistingFileHandling.RAISE,
        data_type_error_handling: ErrorHandling = ErrorHandling.RAISE,
        annotation_error_handling: ErrorHandling = ErrorHandling.RAISE,
        error_handling: ErrorHandling = ErrorHandling.RAISE,
        with_sleep_scoring: bool = False,
    ):
        super().__init__(
            existing_file_handling=existing_file_handling, error_handling=error_handling
        )

        self.input_data_types = input_data_types
        self.output_data_types = output_data_types
        self.pipeline_config = pipeline_config
        self.sample_rate = sample_rate
        self.annotation_type = annotation_type
        self.data_type_error_handling = data_type_error_handling
        self.annotation_error_handling = annotation_error_handling
        self._resample = Resample(sample_rate) if sample_rate is not None else None
        self.with_sleep_scoring = with_sleep_scoring

    def _get_processed_recordings(self, out_dir: Path) -> set[str]:
        """Get set of recording IDs that were successfully processed."""
        catalog_path = out_dir / settings.USLEEP["catalog_file"]
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
                logger.info(f"Skipping recording {recording_id} -" " already processed")
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
                annotation_mapping = (
                    dataset.hypnogram_mapping or settings.DEFAULTS["hynogram_mapping"]
                )
                record_info.update(
                    {f"{label}_count": 0 for label in annotation_mapping.values()}
                )

            try:
                # Extract and catalog annotations
                if self.annotation_type is not None:
                    annotation_info = self._extract_hypnogram(
                        recording, out_dir_path, dataset.hypnogram_mapping
                    )
                    record_info.update(annotation_info)

                # Extract and catalog data types
                data_info = self._extract_data_types(recording, out_dir_path)
                record_info.update(data_info)

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
            ) as e:
                self._handle_error(e, recording)
                record_info["error_message"] = str(e)

            # Update catalog after each recording
            self._update_catalog(record_info, out_dir)

        logger.info(f"Prepared {n_new_processed_recordings} recordings.")

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

        if not self._handle_existing_file(out_file_path):
            return {}

        data_types = recording.read_data_types(self.input_data_types)
        logger.info(f"Read initial data types: {list(data_types.keys())}")

        # Execute pipeline if provided
        if self.pipeline_config:
            data_types = execute_pipeline(
                pipeline_config=self.pipeline_config,
                initial_data=data_types,
                output_data_types=None,  # Get all pipeline outputs
            )
            logger.info(f"Pipeline produced data types: {list(data_types.keys())}")

        # Apply resampling if specified
        if self._resample:
            for data_type in data_types:
                data_types[data_type] = self._resample(data_types[data_type])

        # Write outputs to h5 file
        data_types_info = {}
        extracted_data_types = []
        expected_duration = None
        index = 0

        with h5py.File(out_file_path, "w") as out_file:
            out_file.create_group("channels")

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

                # Calculate and store channel statistics as separate columns
                channel_stats = self._calculate_channel_stats(data)
                for stat_name, stat_value in channel_stats.items():
                    data_types_info[f"{data_type}_{stat_name}"] = stat_value

                self._write_data_to_hdf5(out_file, data_type, data, index)
                extracted_data_types.append(data_type)
                data_types_info[data_type] = True
                index += 1

            logger.info(
                f"Extracted {len(extracted_data_types)} data types for recording "
                f"{recording}: {', '.join(extracted_data_types)}"
            )

            if self.sample_rate is not None:
                out_file.attrs["sample_rate"] = self.sample_rate

        return data_types_info

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

    def _update_catalog(self, record_info: dict, out_dir: Path) -> None:
        """Update catalog with new record info."""
        catalog_path = out_dir / settings.USLEEP["catalog_file"]

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
