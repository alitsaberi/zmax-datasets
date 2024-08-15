import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from zmax_datasets import settings
from zmax_datasets.datasets.exceptions import SleepScoringReadError
from zmax_datasets.datasets.utils import (
    extract_id_by_regex,
)
from zmax_datasets.datasets.zmax import HandlingStrategy, ZMaxDataset, ZMaxRecording
from zmax_datasets.utils.helpers import load_yaml_config
from zmax_datasets.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class Donders2022(ZMaxDataset):
    _SUBJECT_DIR_REGEX: re.Pattern = re.compile(r"s(?P<id>\d+)")
    _SESSION_DIR_REGEX: re.Pattern = re.compile(r"n(?P<id>\d+)")
    _ZMAX_DIR_NAME: str = "zmax"
    _MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN: str = (
        "s{subject_id:d} n{session_id:d}_psg.txt"
    )
    _MANUAL_SLEEP_SCORES_FILE_SEPARATORS: list[str] = [" ", "\t"]

    def _collect_recordings(self) -> list[ZMaxRecording]:
        return [
            recording
            for zmax_dir in self.data_dir.rglob(f"**/{self._ZMAX_DIR_NAME}")
            if (recording := self._process_zmax_dir(zmax_dir)) is not None
        ]

    def _process_zmax_dir(self, zmax_dir: Path) -> ZMaxRecording | None:
        subject_id, session_id = self._extract_ids_from_zmax_dir(zmax_dir)

        if subject_id is None or session_id is None:
            logger.debug(
                "Skipping recording with because"
                f"Could not extract subject and session IDs from {zmax_dir}"
            )
            return

        if not self._is_recording_included(subject_id, session_id):
            logger.debug(
                "Skipping recording with because it was excluded:"
                f" {subject_id}-{session_id}"
            )
            return

        return ZMaxRecording(
            subject_id=subject_id,
            session_id=session_id,
            data_dir=zmax_dir,
            sleep_scores_file=self._get_sleep_scores_file(
                zmax_dir, subject_id, session_id
            ),
        )

    @classmethod
    def _extract_ids_from_zmax_dir(
        cls, zmax_dir: Path
    ) -> tuple[int | None, int | None]:
        subject_id = extract_id_by_regex(
            zmax_dir.parent.parent.name, cls._SUBJECT_DIR_REGEX
        )
        session_id = extract_id_by_regex(zmax_dir.parent.name, cls._SESSION_DIR_REGEX)
        return subject_id, session_id

    def _get_sleep_scores_file(
        self, zmax_dir: Path, subject_id: int, session_id: int
    ) -> Path | None:
        sleep_scores_file = (
            zmax_dir
            / self._MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN.format(
                subject_id=subject_id,
                session_id=session_id,
            )
        )
        return sleep_scores_file if sleep_scores_file.is_file() else None

    @classmethod
    def _read_hypnogram(cls, hypnogram_file: Path) -> np.ndarray:
        return cls._read_manual_sleep_scores(hypnogram_file)[
            "sleep_stage"
        ].values.squeeze()

    @classmethod
    def _read_manual_sleep_scores(cls, sleep_scores_file: Path) -> pd.DataFrame:
        for separator in cls._MANUAL_SLEEP_SCORES_FILE_SEPARATORS:
            try:
                return pd.read_csv(
                    sleep_scores_file,
                    sep=separator,
                    names=["sleep_stage", "arousal"],
                    dtype=int,
                )
            except ValueError:
                logger.debug(
                    f"Failed to read sleep scores file {sleep_scores_file}"
                    f" with separator {separator}. Trying next separator"
                )

        raise SleepScoringReadError(
            f"Failed to read hypnogram file {sleep_scores_file} with default separators"
            f" {cls._MANUAL_SLEEP_SCORES_FILE_SEPARATORS}"
        )


if __name__ == "__main__":
    config_file = settings.CONFIG_DIR / "donders_2022.yaml"
    setup_logging()
    config = load_yaml_config(config_file)
    dataset = Donders2022(**config["datasets"]["donders_2022"])
    print(len(dataset))
    dataset.to_usleep(
        out_dir=settings.DATA_DIR / "donders_2022",
        data_types=["EEG R", "EEG L"],
        data_type_labels={
            "EEG L": "F7-Fpz",
            "EEG R": "F8-Fpz",
        },
        existing_file_handling=HandlingStrategy.SKIP,
        missing_data_type_handling=HandlingStrategy.SKIP,
    )
    # dataset.preprare_recordings()
    # print(dataset.recordings)
    # print(dataset[0])
