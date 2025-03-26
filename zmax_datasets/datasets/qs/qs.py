import logging
from functools import cached_property
from pathlib import Path

import pandas as pd

from zmax_datasets import settings
from zmax_datasets.datasets.base import (
    DataTypeMapping,
    ZMaxDataset,
    ZMaxRecording,
)
from zmax_datasets.exports.usleep import (
    ErrorHandling,
    ExistingFileHandling,
    USleepExportStrategy,
)
from zmax_datasets.utils.exceptions import (
    MultipleSleepScoringFilesFoundError,
    SleepScoringFileNotFoundError,
)
from zmax_datasets.utils.helpers import load_yaml_config
from zmax_datasets.utils.logger import setup_logging
from zmax_datasets.utils.transforms import (
    clip_noisy_values,
    extract_hrv,
    fir_filter,
    l2_normalize,
)

logger = logging.getLogger(__name__)


_SCORING_MAPPING_FILE = Path(__file__).parent / "qs_scoring_files.csv"
_SCORING_MAPPING_FILE_COLUMNS = ["session_id", "scoring_file"]
_SUBJECT_ID = "s1"


class QS(ZMaxDataset):
    @cached_property
    def _scoring_mapping(self) -> pd.DataFrame:
        return pd.read_csv(
            _SCORING_MAPPING_FILE,
            names=_SCORING_MAPPING_FILE_COLUMNS,
        )

    @classmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        return _SUBJECT_ID, zmax_dir.name

    def _get_sleep_scoring_file(self, recording: ZMaxRecording) -> Path:
        matching_rows = self._scoring_mapping[
            self._scoring_mapping["session_id"] == recording.session_id
        ]

        if matching_rows.empty:
            raise SleepScoringFileNotFoundError(
                f"No scoring file found for {recording}."
            )

        if (scoring_files_count := len(matching_rows)) > 1:
            raise MultipleSleepScoringFilesFoundError(
                f"Multiple scoring files ({scoring_files_count}) found for {recording}."
            )

        return self._sleep_scoring_dir / matching_rows["scoring_file"].iloc[0]


if __name__ == "__main__":
    setup_logging()
    config_file = settings.CONFIG_DIR / "datasets.yaml"
    config = load_yaml_config(config_file)
    dataset = QS(**config["QS"])
    sampling_frequency = settings.ZMAX["sampling_frequency"]
    export_strategy = USleepExportStrategy(
        data_type_mappigns=[
            DataTypeMapping(
                "F7-Fpz",
                ["EEG L"],
                transforms=[
                    (
                        fir_filter,
                        {
                            "sampling_frequency": sampling_frequency,
                            "low_cutoff": 0.3,
                            "high_cutoff": 30,
                        },
                    ),
                    (
                        clip_noisy_values,
                        {
                            "min_max_times_global_iqr": 20,
                        },
                    ),
                ],
            ),
            DataTypeMapping(
                "F8-Fpz",
                ["EEG R"],
                transforms=[
                    (
                        fir_filter,
                        {
                            "sampling_frequency": sampling_frequency,
                            "low_cutoff": 0.3,
                            "high_cutoff": 30,
                        },
                    ),
                    (
                        clip_noisy_values,
                        {
                            "min_max_times_global_iqr": 20,
                        },
                    ),
                ],
            ),
            DataTypeMapping(
                "movement",
                ["dX", "dY", "dZ"],
                transforms=[
                    (
                        fir_filter,
                        {
                            "sampling_frequency": sampling_frequency,
                            "high_cutoff": 5,
                        },
                    ),
                    l2_normalize,
                ],
            ),
            DataTypeMapping(
                "heart_rate_variability",
                ["OXY_IR_AC"],
                transforms=[
                    (
                        fir_filter,
                        {
                            "sampling_frequency": sampling_frequency,
                            "low_cutoff": 0.5,
                            "high_cutoff": 4,
                        },
                    ),
                    (
                        extract_hrv,
                        {
                            "sampling_frequency": sampling_frequency,
                            "distance": 0.5,
                            "sliding_window_length": 10,
                            "interpolate": True,
                        },
                    ),
                ],
            ),
        ],
        existing_file_handling=ExistingFileHandling.OVERWRITE,
        error_handling=ErrorHandling.SKIP,
    )
    export_strategy.export(dataset, Path("/project/3013102.01/sleep_scoring/qs"))
    # export_strategy = YasaExportStrategy(
    #     eeg_channel="EEG L",
    #     eog_channel="EEG R",
    #     sampling_frequency=100,
    #     test_split_size=0.0,
    #     existing_file_handling=ExistingFileHandling.APPEND,
    #     error_handling=ErrorHandling.SKIP,
    # )
    # export_strategy.export(dataset, Path("data/yasa"))
