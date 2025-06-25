from pathlib import Path

import numpy as np
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.datasets.base import Dataset, DataTypeMapping, Recording
from zmax_datasets.exports.base import ExportStrategy
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.processing.artifact_detection import get_usability_scores, load_model
from zmax_datasets.utils.helpers import remove_tree


class ArtifactExportStrategy(ExportStrategy):
    def __init__(
        self,
        data_type_mappings: list[DataTypeMapping],
        existing_file_handling: ExistingFileHandling = ExistingFileHandling.RAISE_ERROR,
        error_handling: ErrorHandling = ErrorHandling.RAISE,
    ):
        if len(data_type_mappings) != 3:
            raise ValueError(
                "Artifact export strategy requires exactly 3 data type mappings"
                " in the following order: EEG left, EEG right, movement, but got"
                f" {len(data_type_mappings)}"
            )

        super().__init__(
            existing_file_handling=existing_file_handling, error_handling=error_handling
        )
        (
            self.eeg_left_data_type_mapping,
            self.eeg_right_data_type_mapping,
            self.movement_data_type_mapping,
        ) = data_type_mappings
        self._model = load_model()

    def _export(self, dataset: Dataset, out_dir: Path) -> None:
        prepared_recordings = 0
        for i, recording in enumerate(dataset.get_recordings(with_sleep_scoring=True)):
            logger.info(f"-> Recording {i+1}: {recording}")

            recording_out_dir = out_dir / str(recording)

            if not recording_out_dir.exists():
                logger.warning(
                    f"Skipping recording {recording} because output directory"
                    f" {recording_out_dir} does not exist"
                )
                continue

            try:
                self._extract_artifact_labels(recording, recording_out_dir)
                prepared_recordings += 1
            except (FileExistsError, FileNotFoundError) as e:
                self._handle_error(e, recording, recording_out_dir)

        logger.info(f"Prepared {prepared_recordings} recordings for Artifact Detection")

    def _extract_artifact_labels(
        self, recording: Recording, recording_out_dir: Path
    ) -> None:
        logger.info("Extracting artifact labels...")

        out_file_path = (
            recording_out_dir
            / f"{recording}.{settings.ARTIFACT_DETECTION['labels_file_extension']}"
        )
        self._check_existing_file(out_file_path)

        # Assuming data_type_mappings[0] and [1] are EEG, [2] is movement
        eeg_left = self.eeg_left_data_type_mapping.map(
            recording, settings.ARTIFACT_DETECTION["sampling_frequency"]
        )
        eeg_right = self.eeg_right_data_type_mapping.map(
            recording, settings.ARTIFACT_DETECTION["sampling_frequency"]
        )
        movement = self.movement_data_type_mapping.map(
            recording, settings.ARTIFACT_DETECTION["sampling_frequency"]
        )

        # Combine EEG and movement data
        combined_data = np.vstack((eeg_left, eeg_right, movement)).T
        logger.debug(f"Combined data shape: {combined_data.shape}")

        usability_scores, _, _ = get_usability_scores(
            combined_data,
            settings.ARTIFACT_DETECTION["sampling_frequency"],
            self._model,
        )

        with open(out_file_path, "w") as out_file:
            out_file.write(
                f"{self.eeg_left_data_type_mapping.output_label},{self.eeg_right_data_type_mapping.output_label}\n"
            )
            for score in usability_scores:
                out_file.write(f"{score[0]},{score[1]}\n")

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
