import logging
import re
from collections.abc import Callable
from itertools import chain
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from zmax_real_time.datasets.utils import extract_id_by_regex, rescale_and_clip_data
from zmax_real_time.utils import load_yaml_config

logger = logging.getLogger(__name__)  # TODO: configure logger


class Donders2022(Dataset):
    _SUBJECT_DIR_REGEX = re.compile(r"s(?P<id>\d+)")
    _SESSION_DIR_REGEX = re.compile(r"n(?P<id>\d+)")
    _ZMAX_DIR_NAME = "zmax"
    _MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN = "s{subject_id:d} n{session_id:d}_psg.txt"

    def __init__(
        self,
        data_dir: Path | str,
        channel_names: list[str] | None = None,
        annotations: list[str] | None = None,
        selected_subjects: list[int] | None = None,
        selected_sessions: list[int] | None = None,
        excluded_subjects: list[int] | None = None,
        excluded_sessions: list[int] | None = None,
        excluded_sessions_for_subjects: dict[int, list[int]] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.channel_names = channel_names or ["EEG L", "EEG R"]
        self.annotations = annotations or ["sleep_stage"]
        self.selected_subjects = selected_subjects
        self.selected_sessions = selected_sessions
        self.excluded_subjects = excluded_subjects
        self.excluded_sessions = excluded_sessions
        self.excluded_sessions_for_subjects = excluded_sessions_for_subjects
        self.transform = transform

        self.recordings = self._collect_recordings()
        # TODO: prepare recordings by creating a file

    def _collect_recordings(self) -> list[dict]:
        recordings = []

        for zmax_dir in self.data_dir.rglob(f"**/{self._ZMAX_DIR_NAME}"):
            subject_id, session_id = self._extract_ids_from_zmax_dir(zmax_dir)
            annotation_file = (
                zmax_dir
                / self._MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN.format(
                    subject_id=subject_id, session_id=session_id
                )
            )

            if subject_id is None or session_id is None:
                logger.debug(
                    "Skipping recording with because"
                    f" could not extract subject and session IDs from {zmax_dir}"
                )
                continue

            if not self._is_recording_included(subject_id, session_id):
                logger.debug(
                    "Skipping recording with because it was excluded:"
                    f" {subject_id}-{session_id}"
                )
                continue

            if any(
                not file.exists()
                for file in chain(
                    (zmax_dir / f"{channel}.edf" for channel in self.channel_names),
                    [annotation_file],
                )
            ):
                logger.debug(
                    "Skipping recording with because some files are missing:"
                    " {subject_id}-{session_id}"
                )
                continue

            recordings.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "zmax_dir": zmax_dir,
                }
            )

        return recordings

    @classmethod
    def _extract_ids_from_zmax_dir(
        cls, zmax_dir: Path
    ) -> tuple[int | None, int | None]:
        subject_id = extract_id_by_regex(
            zmax_dir.parent.parent.name, cls._SUBJECT_DIR_REGEX
        )
        session_id = extract_id_by_regex(zmax_dir.parent.name, cls._SESSION_DIR_REGEX)
        return subject_id, session_id

    def _is_recording_included(self, subject_id: int, session_id: int) -> bool:
        return (
            (not self.selected_subjects or subject_id in self.selected_subjects)
            and (not self.excluded_subjects or subject_id not in self.excluded_subjects)
            and (not self.selected_sessions or session_id in self.selected_sessions)
            and (not self.excluded_sessions or session_id not in self.excluded_sessions)
            and (
                not self.excluded_sessions_for_subjects
                or subject_id not in self.excluded_sessions_for_subjects
                or session_id not in self.excluded_sessions_for_subjects[subject_id]
            )
        )

    def _prepare_recording(self, recording):
        channel_data = []
        times = None
        for channel in self.channel_names:
            channel_file = recording["zmax_dir"] / f"{channel}.edf"
            raw = mne.io.read_raw_edf(channel_file, preload=True)
            raw = self._preprocess_channel(raw)

            if times is None:
                times = raw.times
            elif not np.array_equal(times, raw.times):
                raise ValueError(
                    f"Time mismatch for {channel} in recording"
                    f" {recording['subject_id']}-{recording['session_id']}"
                )

            channel_data.append(raw.get_data())

        # Combine all channels
        combined_data = np.concatenate(channel_data, axis=0)
        recording["channel_data"] = combined_data
        recording["times"] = times

        annotations = pd.read_csv(
            recording["zmax_dir"]
            / self._MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN.format(
                subject_id=recording["subject_id"], session_id=recording["session_id"]
            ),
            delimiter=" ",
            names=["sleep_stage", "arousal"],
            usecols=self.annotations,
        )

        print(annotations)

    def _preprocess_channel(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw.filter(l_freq=0.3, h_freq=30)
        raw.resample(128)
        raw.notch_filter(freqs=50)
        raw.apply_function(lambda x: rescale_and_clip_data(x))
        return raw

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        recording = self.recordings[idx]
        self._prepare_recording(recording)

        # # Get corresponding sleep score
        # sleep_score = self.scores_df.loc[
        #     self.scores_df["filename"] == eeg_file, "score"
        # ].values[0]

        # # Convert to PyTorch tensors
        # eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        # sleep_score = torch.tensor(sleep_score, dtype=torch.float32)

        # if self.transform:
        #     eeg_data = self.transform(eeg_data)

        # return eeg_data, sleep_score


if __name__ == "__main__":
    CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "donders_2022.yaml"
    config = load_yaml_config(CONFIG_PATH)
    print(config)
    dataset = Donders2022(**config["datasets"]["donders_2022"])
    print(len(dataset))
    print(dataset.recordings)
    print(dataset[0])
