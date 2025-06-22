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
