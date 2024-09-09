import logging
from pathlib import Path

import mne
import numpy as np
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
    ChannelLengthMismatchError,
    InvalidZMaxDataTypeError,
    MissingDataTypesError,
    NoFeaturesExtractedError,
    SleepScoringReadError,
)

logger = logging.getLogger(__name__)


_OUT_FILE_NAME = "zmax.parquet"
_DATA_FRAME_INDICES = ["dataset", "subject_id", "session_id", "epoch"]


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

        try:
            df = self._load_existing_data(out_file_path)
            hypnogram_mapping = self._update_hypnogram_mapping(
                dataset.hypnogram_mapping
            )
            dataset_features = self._process_recordings(dataset, hypnogram_mapping)
            df = self._update_dataframe(
                df, dataset_features, dataset.__class__.__name__
            )
            df.to_parquet(out_file_path)
        except NoFeaturesExtractedError as e:
            self._handle_error(e)

        logger.info(f"Prepared {len(dataset_features)} recordings for Yasa.")
        logger.info(f"Created {out_file_path} with {df}.")

    def _load_existing_data(self, file_path: Path) -> pd.DataFrame:
        if file_path.exists():
            if self.existing_file_handling == ExistingFileHandling.RAISE_ERROR:
                raise FileExistsError(f"File {file_path} already exists.")

            logger.info(f"File {file_path} already exists.")
            if self.existing_file_handling == ExistingFileHandling.APPEND:
                df = pd.read_parquet(file_path)
                logger.info(
                    f"Loaded existing data from {file_path}."
                    f" Datasets: {df.index.unique(level=0).to_list()}"
                )
                return df
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
    ) -> list[pd.DataFrame]:
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
                ChannelLengthMismatchError,
            ) as e:
                self._handle_error(e, f"Skipping recording {recording}")
        return dataset_features

    def _extract_features(
        self,
        recording: ZMaxRecording,
    ) -> pd.DataFrame:
        combined_raw = self._create_combined_raw(recording)

        yasa_feature_extractor = yasa.SleepStaging(
            combined_raw,
            eeg_name=self.eeg_channel,
            eog_name=self.eog_channel,
        )

        features = yasa_feature_extractor.get_features().reset_index()
        features["subject_id"] = recording.subject_id
        features["session_id"] = recording.session_id

        return features

    def _create_combined_raw(self, recording: ZMaxRecording) -> mne.io.Raw:
        raw_eeg = self._read_raw_data(recording, self.eeg_channel)

        if not self.eog_channel:
            return raw_eeg

        raw_eog = self._read_raw_data(recording, self.eog_channel)

        eeg_data, _ = raw_eeg[:]
        eog_data, _ = raw_eog[:]

        if len(eeg_data) != len(eog_data):
            raise ChannelLengthMismatchError(
                "EEG and EOG channels have different lengths."
                f" EEG: {len(eeg_data)}, EOG: {len(eog_data)}"
            )

        combined_data = np.vstack([eeg_data, eog_data])

        channel_names = raw_eeg.ch_names + raw_eog.ch_names
        channel_types = ["eeg", "eog"]

        combined_info = mne.create_info(
            ch_names=channel_names, sfreq=raw_eeg.info["sfreq"], ch_types=channel_types
        )

        return mne.io.RawArray(combined_data, combined_info)

    def _read_raw_data(self, recording: ZMaxRecording, data_type: str) -> mne.io.Raw:
        raw = recording.read_raw_data(data_type)
        raw.resample(self.sampling_frequency, npad="auto")
        return raw

    def _handle_error(self, error: Exception, message: str | None = None) -> None:
        if self.error_handling == ErrorHandling.SKIP:
            log_message = f"{message}: {error}" if message is not None else str(error)
            logger.warning(log_message)
        elif self.error_handling == ErrorHandling.RAISE:
            raise error

    def _update_dataframe(
        self, df: pd.DataFrame, dataset_features: list, dataset_name: str
    ) -> pd.DataFrame:
        if not dataset_features:
            raise NoFeaturesExtractedError(
                f"No features extracted for dataset {dataset_name}"
            )

        df_ = pd.concat(dataset_features)
        df_["dataset"] = dataset_name
        df_["dataset"] = df_["dataset"].astype("category")
        df_["stage"] = df_["stage"].astype("category")
        df_.set_index(_DATA_FRAME_INDICES, inplace=True)
        logger.debug(f"Dataset features: {df_}")
        return df_.combine_first(df)
