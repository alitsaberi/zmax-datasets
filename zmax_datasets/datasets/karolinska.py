import logging
from pathlib import Path

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
from zmax_datasets.utils.helpers import load_yaml_config
from zmax_datasets.utils.logger import setup_logging
from zmax_datasets.utils.transforms import (
    clip_noisy_values,
    extract_hrv,
    fir_filter,
    l2_normalize,
)

logger = logging.getLogger(__name__)


class Karolinska(ZMaxDataset):
    @classmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        return zmax_dir.parent.parent.name, zmax_dir.name

    def _get_sleep_scoring_file(self, recording: ZMaxRecording) -> Path:
        return self._sleep_scoring_dir / self._sleep_scoring_file_pattern.format(
            subject_id=recording.subject_id.replace("SNZ_", ""),
            session_id=recording.session_id,
        )


if __name__ == "__main__":
    setup_logging()
    config_file = settings.CONFIG_DIR / "datasets.yaml"
    config = load_yaml_config(config_file)
    dataset = Karolinska(**config["datasets"]["karolinska"])
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
    export_strategy.export(dataset, Path("data/karolinska"))
    # export_strategy = YasaExportStrategy(
    #     eeg_channel="EEG L",
    #     eog_channel="EEG R",
    #     sampling_frequency=100,
    #     test_split_size=0.2,
    #     existing_file_handling=ExistingFileHandling.APPEND,
    #     error_handling=ErrorHandling.SKIP,
    # )
    # export_strategy.export(dataset, Path("data/yasa"))
