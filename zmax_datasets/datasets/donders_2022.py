import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from psg_utils.hypnogram.utils import squeeze_events

from zmax_datasets import settings
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

    def _extract_hypnogram(self, recording: dict):
        annotations = pd.read_csv(
            recording["zmax_dir"]
            / self._MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN.format(
                subject_id=recording["subject_id"],
                session_id=recording["session_id"],
            ),
            delimiter="\t",
            names=["sleep_stage", "arousal"],
            usecols=self.annotations,
        )

        map_ = np.vectorize(
            {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", -1: "UNKNOWN"}.get
        )

        stages = map_(annotations["sleep_stage"].values)
        stages = stages.squeeze()

        inits, durs, stages = self.ndarray_to_ids_format(
            stages=stages,
            period_length=30,
        )
        inits, durs, stages = squeeze_events(inits, durs, stages)

        out_dir = Path(
            settings.BASE_DIR
            / "data"
            / "donders_2022"
            / f"s{recording['subject_id']}-n{recording['session_id']}"
        )
        out = out_dir / f"s{recording['subject_id']}-n{recording['session_id']}.ids"
        with open(out, "w") as out_f:
            for i, d, s in zip(inits, durs, stages, strict=False):
                out_f.write(f"{int(i)},{int(d)},{s}\n")
        return zmax_dir / self._MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN

    def ndarray_to_ids_format(self, stages, period_length):
        start_times = []
        durations = []

        start_time = 0
        # Process each sleep stage
        for _ in stages:
            # Create a row with start time, duration, and sleep stage
            start_times.append(start_time)

            # Append the duration to the durations array (all durations are 30 seconds)
            durations.append(period_length)

            # Increment start_time by the epoch duration
            start_time += period_length

        return (
            np.asarray(start_times, float),
            np.asarray(durations, float),
            stages,
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
        existing_file_handling=HandlingStrategy.OVERWRITE,
        missing_data_type_handling=HandlingStrategy.SKIP,
    )
    # dataset.preprare_recordings()
    # print(dataset.recordings)
    # print(dataset[0])
