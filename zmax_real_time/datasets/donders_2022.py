import logging
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from zmax_real_time.utils import load_yaml_config

logger = logging.getLogger(__name__)  # TODO: configure logger


class Donders2022(Dataset):
    _ID_GROUP_NAME = "id"
    _SUBJECT_NAME_PATTERN = "s*"
    _SESSION_NAME_PATTERN = "n*"
    _ZMAX_DIR_NAME = "zmax"
    _MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN = "*_psg.txt"
    _MANUAL_SLEEP_SCORES_FILE_IDS_REGEX = re.compile(
        r"s(?P<subject_id>\d+) n(?P<session_id>\d+)"
    )

    def __init__(
        self,
        data_dir: Path | str,
        selected_subjects: list[int] | None = None,
        selected_sessions: list[int] | None = None,
        excluded_subjects: list[int] | None = None,
        excluded_sessions: list[int] | None = None,
        excluded_sessions_for_subjects: dict[int, list[int]] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.selected_subjects = selected_subjects
        self.selected_sessions = selected_sessions
        self.excluded_subjects = excluded_subjects
        self.excluded_sessions = excluded_sessions
        self.excluded_sessions_for_subjects = excluded_sessions_for_subjects
        self.transform = transform

        self.recordings = self._get_recordings()

    def _get_recordings(self) -> pd.DataFrame:
        recordings = []

        for manual_sleep_scores_file in self.data_dir.rglob(
            f"{self._SUBJECT_NAME_PATTERN}/{self._SESSION_NAME_PATTERN}/{self._ZMAX_DIR_NAME}/{self._MANUAL_SLEEP_SCORES_FILE_NAME_PATTERN}"
        ):
            subject_id, session_id = self._extract_ids_from_file_name(
                manual_sleep_scores_file.name
            )

            if subject_id is None:
                logger.debug(
                    "Failed to extract IDs from file name:"
                    f" {manual_sleep_scores_file.name}"
                )
                continue

            recordings.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "manual_sleep_scores_file": manual_sleep_scores_file,
                }
            )

        recordings = pd.DataFrame(recordings)
        recordings = self._filter_recordings(recordings)

        return recordings

    @classmethod
    def _extract_ids_from_file_name(
        cls, file_name: str
    ) -> tuple[int | None, int | None]:
        match = re.search(cls._MANUAL_SLEEP_SCORES_FILE_IDS_REGEX, file_name)
        if match:
            try:
                return int(match.group("subject_id")), int(match.group("session_id"))
            except ValueError as e:
                logger.debug(f"ValueError in parsing IDs: {e}: {file_name}")

        return None, None

    def _filter_recordings(self, recordings: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series([True] * len(recordings), index=recordings.index)

        if self.selected_subjects is not None:
            mask &= recordings["subject_id"].isin(self.selected_subjects)

        if self.excluded_subjects is not None:
            mask &= ~recordings["subject_id"].isin(self.excluded_subjects)

        if self.selected_sessions is not None:
            mask &= recordings["session_id"].isin(self.selected_sessions)

        if self.excluded_sessions is not None:
            mask &= ~recordings["session_id"].isin(self.excluded_sessions)

        if self.excluded_sessions_for_subjects is not None:
            mask &= ~recordings.apply(
                lambda row: row["subject_id"] in self.excluded_sessions_for_subjects
                and row["session_id"]
                in self.excluded_sessions_for_subjects[row["subject_id"]],
                axis=1,
            )

        return recordings[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        recording = self.recordings.iloc[idx]
        sleep_scores = pd.read_csv(
            recording["manual_sleep_scores_file"],
            delimiter=" ",
            names=["sleep_stage", "arousal"],
        )
        return sleep_scores

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
    dataset = Donders2022(**config)
    print(len(dataset))
    print(dataset.recordings)
    print(dataset[0])
