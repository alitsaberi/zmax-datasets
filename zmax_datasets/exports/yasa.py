from pathlib import Path
from random import sample

import mne
import numpy as np
import pandas as pd
import yasa
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.datasets.base import (
    Dataset,
    Recording,
)
from zmax_datasets.exports.base import ExportStrategy
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.sources.zmax.enums import DataTypes
from zmax_datasets.utils.exceptions import (
    ChannelLengthMismatchError,
    HypnogramMismatchError,
    InvalidZMaxDataTypeError,
    MissingDataTypeError,
    NoFeaturesExtractedError,
    SleepScoringReadError,
)

_OUT_FILE_NAME = "zmax.parquet"
_DATASET_COLUMN = "dataset"
_SUBJECT_COLUMN = "subj"
_SESSION_COLUMN = "session"
_SPLIT_COLUMN = "set"
_DATA_FRAME_INDICES = [_SPLIT_COLUMN, _DATASET_COLUMN, _SUBJECT_COLUMN, _SESSION_COLUMN]


def summarize_dataset(df: pd.DataFrame) -> None:
    summary_data = []
    splits = list(settings.YASA["split_labels"].keys()) + ["total"]

    for split in splits:
        split_df = (
            df
            if split == "total"
            else df[
                df.index.get_level_values(_SPLIT_COLUMN)
                == settings.YASA["split_labels"][split]
            ]
        )

        summary_data.append(
            {
                "num_subjects": split_df.index.get_level_values(
                    _SUBJECT_COLUMN
                ).nunique(),
                "num_recordings": split_df.groupby(
                    [_SUBJECT_COLUMN, _SESSION_COLUMN]
                ).ngroups,
                "num_samples": len(split_df),
            }
        )

    summary_df = pd.DataFrame(summary_data, index=splits)
    return summary_df


class YasaExportStrategy(ExportStrategy):
    def __init__(
        self,
        eeg_channel: str,
        eog_channel: str | None = None,
        sampling_frequency: int = settings.YASA["sampling_frequency"],
        test_split_size: float = 0.2,
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
        self.test_split_size = test_split_size

    def _validate_channels(self, eeg_channel: str, eog_channel: str | None) -> None:
        eeg_data_types = DataTypes.get_by_category("EEG")
        eeg_data_type = DataTypes.get_by_channel(eeg_channel)
        eog_data_type = DataTypes.get_by_channel(eog_channel) if eog_channel else None

        if eeg_data_type is None or eeg_data_type not in eeg_data_types:
            print("hi")
            print(eeg_data_type)
            raise InvalidZMaxDataTypeError(
                f"{eeg_channel} is not a valid ZMax EEG channel."
                " Valid channels are:"
                f" {[data_type.channel for data_type in eeg_data_types]}"
            )

        if eog_data_type is None or eog_data_type not in eeg_data_types:
            raise InvalidZMaxDataTypeError(
                f"{eog_channel} is not a valid ZMax EOG channel."
                " Valid channels are:"
                f" {[data_type.channel for data_type in eeg_data_types]}"
            )

    def _export(self, dataset: Dataset, out_dir: Path) -> None:
        out_file_path = out_dir / _OUT_FILE_NAME

        try:
            df = self._load_existing_data(out_file_path)
            hypnogram_mapping = self._update_hypnogram_mapping(
                dataset.hypnogram_mapping
            )
            dataset_features = self._process_recordings(dataset, hypnogram_mapping)
            dataset_df = self._create_dataset(
                dataset_features, dataset.__class__.__name__
            )
            logger.info(summarize_dataset(dataset_df))
            df = dataset_df.combine_first(df)
            df.sort_index(inplace=True)
            df.to_parquet(out_file_path)
        except NoFeaturesExtractedError as e:
            self._handle_error(e)

        logger.info(f"Created {out_file_path}.")

    def _load_existing_data(self, file_path: Path) -> pd.DataFrame:
        if file_path.exists():
            if self.existing_file_handling == ExistingFileHandling.RAISE_ERROR:
                raise FileExistsError(f"File {file_path} already exists.")

            logger.info(f"File {file_path} already exists.")
            if self.existing_file_handling == ExistingFileHandling.APPEND:
                df = pd.read_parquet(file_path)
                logger.info(
                    f"Loaded existing data from {file_path}."
                    f" Datasets: {df.index.unique(level=1).to_list()}"
                )
                return df
        return pd.DataFrame()

    def _update_hypnogram_mapping(
        self, hypnogram_mapping: dict[int, str]
    ) -> dict[int, str]:
        return {
            key: settings.YASA["hypnogram_mapping"].get(
                value,
                settings.YASA["hypnogram_mapping"][settings.DEFAULTS["label"]],
            )
            for key, value in hypnogram_mapping.items()
        }

    def _process_recordings(
        self, dataset: Dataset, hypnogram_mapping: dict
    ) -> list[pd.DataFrame]:
        dataset_features = []
        for i, recording in enumerate(dataset.get_recordings(with_sleep_scoring=True)):
            logger.info(f"-> Recording {i+1}: {recording}")
            try:
                features = self._extract_features(recording)
                hypnogram = recording.read_annotations(label_mapping=hypnogram_mapping)

                if len(features) != len(hypnogram):
                    raise HypnogramMismatchError(len(features), len(hypnogram))

                features[settings.YASA["hypnogram_column"]] = hypnogram
                dataset_features.append(features)
            except (
                MissingDataTypeError,
                SleepScoringReadError,
                FileNotFoundError,
                HypnogramMismatchError,
                ChannelLengthMismatchError,
            ) as e:
                self._handle_error(e, f"Skipping recording {recording}")

        return dataset_features

    def _extract_features(
        self,
        recording: Recording,
    ) -> pd.DataFrame:
        combined_raw = self._create_combined_raw(recording)

        yasa_feature_extractor = yasa.SleepStaging(
            combined_raw,
            eeg_name=self.eeg_channel,
            eog_name=self.eog_channel,
        )

        features = yasa_feature_extractor.get_features().reset_index()
        features[_SUBJECT_COLUMN] = recording.subject_id
        features[_SESSION_COLUMN] = recording.session_id

        return features

    def _create_combined_raw(self, recording: Recording) -> mne.io.Raw:
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

    def _read_raw_data(self, recording: Recording, data_type: str) -> mne.io.Raw:
        raw = recording.read_raw_data(data_type)
        raw.resample(self.sampling_frequency, npad="auto")
        return raw

    def _handle_error(self, error: Exception, message: str | None = None) -> None:
        if self.error_handling == ErrorHandling.SKIP:
            log_message = f"{message}: {error}" if message is not None else str(error)
            logger.warning(log_message)
        elif self.error_handling == ErrorHandling.RAISE:
            raise error

    def _create_dataset(
        self, dataset_features: list[pd.DataFrame], dataset_name: str
    ) -> pd.DataFrame:
        if not dataset_features:
            raise NoFeaturesExtractedError(
                f"No features extracted for dataset {dataset_name}"
            )

        df = pd.concat(dataset_features)
        df[_DATASET_COLUMN] = dataset_name
        df[_DATASET_COLUMN] = df[_DATASET_COLUMN].astype("category")
        df[settings.YASA["hypnogram_column"]] = df[
            settings.YASA["hypnogram_column"]
        ].astype("category")
        df = self._split_dataset(df)
        df.set_index(_DATA_FRAME_INDICES, inplace=True)
        return df

    def _split_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        subject_ids = df[_SUBJECT_COLUMN].unique()
        test_size = int(len(subject_ids) * self.test_split_size)
        test_subjects = set(sample(list(subject_ids), test_size))

        df[_SPLIT_COLUMN] = df[_SUBJECT_COLUMN].map(
            lambda x: settings.YASA["split_labels"]["test"]
            if x in test_subjects
            else settings.YASA["split_labels"]["train"]
        )
        df[_SPLIT_COLUMN] = df[_SPLIT_COLUMN].astype("category")
        return df
