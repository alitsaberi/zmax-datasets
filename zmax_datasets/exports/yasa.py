import logging
from pathlib import Path

import mne
import pandas as pd
import yasa

from zmax_datasets import settings
from zmax_datasets.datasets.base import (
    EEG_CHANNELS,
    ZMaxDataset,
    ZMaxRecording,
    read_hypnogram,
)
from zmax_datasets.exports.base import ExportStrategy
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.utils.exceptions import (
    InvalidZMaxDataTypeError,
    MissingDataTypesError,
    SleepScoringReadError,
)

logger = logging.getLogger(__name__)


_OUT_FILE_NAME = "zmax.parquet"


class HypnogramMismatchError(Exception):
    def __init__(self, features_length: int, hypnogram_length: int):
        self.message = (
            "Features and hypnogram have different lengths:"
            f" {features_length} and {hypnogram_length}"
        )
        super().__init__(self.message)


class YasaExportStrategy(ExportStrategy):
    def __init__(
        self,
        eeg_channel: str,
        eog_channel: str | None = None,
        sampling_frequency: int = settings.YASA["sampling_frequency"],
        existing_file_handling: ExistingFileHandling = ExistingFileHandling.RAISE_ERROR,
        error_handling: ErrorHandling = ErrorHandling.RAISE,
    ):
        super().__init__(
            existing_file_handling=existing_file_handling, error_handling=error_handling
        )
        self._validate_channels(eeg_channel, eog_channel)
        self.eeg_channel = eeg_channel
        self.eog_channel = eog_channel
        self.sampling_frequency = sampling_frequency

    def _validate_channels(self, eeg_channel: str, eog_channel: str | None) -> None:
        if eeg_channel not in EEG_CHANNELS:
            raise InvalidZMaxDataTypeError(
                f"{eeg_channel} is not a valid ZMax EEG channel."
            )
        if eog_channel and eog_channel not in EEG_CHANNELS:
            raise InvalidZMaxDataTypeError(
                f"{eog_channel} is not a valid ZMax EEG channel."
            )

    def _export(self, dataset: ZMaxDataset, out_dir: Path) -> None:
        out_file_path = out_dir / _OUT_FILE_NAME
        df = self._load_existing_data(out_file_path)

        hypnogram_mapping = self._update_hypnogram_mapping(dataset.hypnogram_mapping)
        dataset_features = self._process_recordings(dataset, hypnogram_mapping)

        df = self._update_dataframe(df, dataset_features, dataset.__class__.__name__)
        df.to_parquet(out_file_path)

        logger.info(f"Prepared {len(dataset_features)} recordings for Yasa.")

    def _load_existing_data(self, file_path: Path) -> pd.DataFrame:
        if file_path.exists():
            if self.existing_file_handling == ExistingFileHandling.RAISE_ERROR:
                raise FileExistsError(f"File {file_path} already exists.")
            elif self.existing_file_handling == ExistingFileHandling.APPEND:
                return pd.read_parquet(file_path)
        return pd.DataFrame()

    def _update_hypnogram_mapping(
        self, hypnogram_mapping: dict[int, str]
    ) -> dict[int, str]:
        return {
            key: settings.YASA["hypnogram_mapping"].get(
                value,
                settings.YASA["hypnogram_mapping"][
                    settings.DEFAULTS["hypnogram_label"]
                ],
            )
            for key, value in hypnogram_mapping.items()
        }

    def _process_recordings(
        self, dataset: ZMaxDataset, hypnogram_mapping: dict
    ) -> list:
        dataset_features = []
        for i, recording in enumerate(dataset.get_recordings(with_sleep_scoring=True)):
            logger.info(f"-> Recording {i+1}: {recording}")
            try:
                features = self._extract_features(recording)
                hypnogram = read_hypnogram(
                    recording, hypnogram_mapping=hypnogram_mapping
                )

                if len(features) != len(hypnogram):
                    raise HypnogramMismatchError(len(features), len(hypnogram))

                features["stage"] = hypnogram
                dataset_features.append(features)
            except (
                MissingDataTypesError,
                SleepScoringReadError,
                FileNotFoundError,
                HypnogramMismatchError,
            ) as e:
                self._handle_error(e, recording)
        return dataset_features

    def _extract_features(
        self,
        recording: ZMaxRecording,
    ) -> pd.DataFrame:
        raw_combined = mne.concatenate_raws(
            [
                self._read_raw_data(recording, data_type)
                for data_type in [self.eeg_channel, self.eog_channel]
            ]
        )

        yasa_feature_extractor = yasa.SleepStaging(
            raw_combined,
            eeg_name=self.eeg_channel,
            eog_name=self.eog_channel,
        )

        features = yasa_feature_extractor.get_features().reset_index()
        features["subject_id"] = recording.subject_id
        features["session_id"] = recording.session_id
        features.set_index(["subject_id", "session_id", "epoch"], inplace=True)

        return features

    def _read_raw_data(self, recording: ZMaxRecording, data_type: str) -> mne.io.Raw:
        raw = recording.read_raw_data(data_type)
        raw.resample(self.sampling_frequency, npad="auto")
        return raw

    def _handle_error(self, error: Exception, recording: ZMaxRecording) -> None:
        if self.error_handling == ErrorHandling.SKIP:
            logger.warning(f"Skipping recording {recording}: {error}")
        elif self.error_handling == ErrorHandling.RAISE:
            raise error

    def _update_dataframe(
        self, df: pd.DataFrame, dataset_features: list, dataset_name: str
    ) -> pd.DataFrame:
        df_ = pd.concat(dataset_features)
        df_["dataset"] = dataset_name
        df_["dataset"] = df_["dataset"].astype("category")
        df_["stage"] = df_["stage"].astype("category")
        df_.set_index(["dataset", "subject_id", "session_id", "epoch"], inplace=True)
        return pd.concat([df, df_])
