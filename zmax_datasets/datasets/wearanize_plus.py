import logging
from collections.abc import Generator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from zmax_datasets import settings
from zmax_datasets.datasets.base import DataTypeMapping, SleepAnnotations
from zmax_datasets.datasets.utils import mapper
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.exports.usleep import USleepExportStrategy
from zmax_datasets.utils.helpers import load_yaml_config
from zmax_datasets.utils.logger import setup_logging
from zmax_datasets.utils.transforms import (
    clip_noisy_values,
    extract_hrv,
    fir_filter,
    l2_normalize,
)

logger = logging.getLogger(__name__)

indices = {
    "zmax": "Zmax",
    "psg": "PSG",
    "empatical": "Emp",
    "activepal": "Activepal",
}


@dataclass
class Recording:
    file_path: Path

    @property
    def subject_id(self) -> str:
        return self.file_path.stem

    @cached_property
    def data_frame(self) -> pd.DataFrame:
        return pd.read_parquet(self.file_path)

    @cached_property
    def data_types(self) -> list[str]:
        return self.data_frame.loc[indices["zmax"], "SignalLabel"]

    @cached_property
    def sleep_scores(self) -> np.ndarray:
        return self.data_frame.loc[indices["psg"], "SleepScores"].get("Manual")

    def __str__(self) -> str:
        return f"{self.subject_id}"

    def read_raw_data(self, data_type: str) -> np.ndarray:
        return self.data_frame.loc[indices["zmax"], "SignalData"][data_type].astype(
            np.float64
        )

    def read_annotations(
        self,
        annotation_type: SleepAnnotations = SleepAnnotations.SLEEP_STAGE,
        label_mapping: dict[int, str] | None = None,
        default_label: str = settings.DEFAULTS["label"],
    ) -> np.ndarray:
        if annotation_type == SleepAnnotations.AROUSAL:
            raise ValueError("Arousal annotations are not supported by Wearanize+.")

        annotations = self.sleep_scores
        logger.debug(f"{annotation_type.value} annotations shape: {annotations.shape}")

        if label_mapping is not None:
            annotations = mapper(label_mapping)(annotations, default_label)

        return annotations


@dataclass
class WDataTypeMapping(DataTypeMapping):
    def _get_raw_data(self, recording: Recording) -> np.ndarray:
        data_list = []

        for input_data_type in self.input_data_types:
            data_list.append(recording.read_raw_data(input_data_type))

        return np.vstack(data_list) if len(data_list) > 1 else data_list[0]


class WearanizePlus:
    def __init__(
        self,
        data_dir: Path | str,
        recording_file_pattern: str,
        hypnogram_mapping: dict[int, str] = settings.DEFAULTS["hynogram_mapping"],
    ) -> None:
        self.data_dir = Path(data_dir)
        self._recording_file_pattern = recording_file_pattern
        self.hypnogram_mapping = hypnogram_mapping

    def get_recordings(
        self, with_sleep_scoring: bool = True
    ) -> Generator[Recording, None, None]:
        for recording_file in self._recording_file_generator():
            recording = Recording(recording_file)

            if with_sleep_scoring and recording.sleep_scores is None:
                logger.info(
                    f"Manual sleep scoring not found for {recording}. Skipping."
                )
                continue

            yield recording

    def _recording_file_generator(self) -> Generator[Path, None, None]:
        yield from self.data_dir.glob(self._recording_file_pattern)


if __name__ == "__main__":
    setup_logging()
    config_file = settings.CONFIG_DIR / "datasets.yaml"
    config = load_yaml_config(config_file)
    dataset = WearanizePlus(**config["WearanizePlus"])
    sampling_frequency = settings.ZMAX["sampling_frequency"]
    export_strategy = USleepExportStrategy(
        data_type_mappigns=[
            DataTypeMapping(
                "F7-Fpz",
                ["EEGL"],
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
                ["EEGR"],
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
                ["ACCX", "ACCY", "ACCZ"],
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
    export_strategy.export(
        dataset, Path("/project/3013102.01/sleep_scoring/wearanize_plus")
    )
