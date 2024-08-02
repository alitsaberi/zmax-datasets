import logging
import re
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)  # TODO: configure logger


class Donders2022(Dataset):
    _ID_GROUP_NAME = "id"

    class EntityType(Enum):
        SUBJECT = "subject"
        SESSION = "session"

    def __init__(
        self,
        data_dir: Path,
        subject_regex: str = r"s(?P<id>\d+)",
        session_regex: str = r"n(?P<id>\d+)",
        zmax_dir_name: str = "zmax",
        psg_dir_name: str = "somno",
        selected_subjects: list[int] | None = None,
        selected_sessions: list[int] | None = None,
        excluded_subjects: list[int] | None = None,
        excluded_sessions: list[int] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.subject_regex = re.compile(subject_regex)
        self.session_regex = re.compile(session_regex)
        self.zmax_dir_name = zmax_dir_name
        self.psg_dir_name = psg_dir_name
        self.selected_subjects = selected_subjects
        self.selected_sessions = selected_sessions
        self.excluded_subjects = excluded_subjects
        self.excluded_sessions = excluded_sessions
        self.transform = transform

        for regex_name in ["subject_regex", "session_regex"]:
            regex = getattr(self, regex_name)
            if self._ID_GROUP_NAME not in regex.groupindex:
                raise ValueError(
                    f"{regex_name} must include a group named '{self._ID_GROUP_NAME}'"
                )

        self.session_dirs = self._get_session_dirs()

    def _get_session_dirs(self) -> list[Path]:
        session_dirs = []

        for subject_dir in self.data_dir.iterdir():
            if not self._is_valid_entity_dir(subject_dir, self.EntityType.SUBJECT):
                continue

            for session_dir in subject_dir.iterdir():
                if not self._is_valid_entity_dir(session_dir, self.EntityType.SESSION):
                    continue

                if not (session_dir / self.zmax_dir_name).is_dir():
                    logger.debug(f"{session_dir} doesn't contain {self.zmax_dir_name}.")
                    continue

                if not (session_dir / self.psg_dir_name).is_dir():
                    logger.debug(f"{session_dir} doesn't contain {self.psg_dir_name}.")
                    continue

                session_dirs.append(session_dir)

        return session_dirs

    def _is_valid_entity_dir(self, entity_dir: Path, entity_type: EntityType) -> bool:
        if not entity_dir.is_dir():
            logger.debug(f"{entity_dir} is not a directory.")
            return False

        entity_regex = getattr(self, f"{entity_type.value}_regex")
        entity_id = self._get_entity_id(entity_dir, entity_regex)

        if entity_id is None:
            logger.debug(f"{entity_dir} doesn't match the regex {entity_regex}.")
            return False

        selected = getattr(self, f"selected_{entity_type.value}s", None)
        if selected and entity_id not in selected:
            logger.debug(f"{entity_dir} is not selected.")
            return False

        excluded = getattr(self, f"excluded_{entity_type.value}s", None)
        if excluded and entity_id in excluded:
            logger.debug(f"{entity_dir} is excluded.")
            return False

        return True

    @classmethod
    def _get_entity_id(cls, entity_dir: Path, regex: re.Pattern) -> int | None:
        return (
            int(match.group(cls._ID_GROUP_NAME))
            if (match := regex.match(entity_dir.name))
            else None
        )

    def __len__(self):
        return len(self.session_dirs_)

    def __getitem__(self, idx):
        # Get EEG file name and load EEG data
        # eeg_file = self.eeg_files[idx]
        # eeg_path = os.path.join(self.eeg_dir, eeg_file)
        # eeg_data = np.load(eeg_path)  # Assuming EEG data is in .npy format

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
        pass


if __name__ == "__main__":
    data_dir = Path("/project/3013097.06/Data")
    dataset = Donders2022(data_dir)
    print(len(dataset))
