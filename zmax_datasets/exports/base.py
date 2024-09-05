import logging
from abc import ABC, abstractmethod
from pathlib import Path

from zmax_datasets.datasets.base import ZMaxDataset
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling

logger = logging.getLogger(__name__)


class ExportStrategy(ABC):
    def __init__(
        self,
        existing_file_handling: ExistingFileHandling = ExistingFileHandling.RAISE_ERROR,
        error_handling: ErrorHandling = ErrorHandling.RAISE,
    ):
        self.existing_file_handling = existing_file_handling
        self.error_handling = error_handling

    def export(self, dataset: ZMaxDataset, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting dataset using {self.__class__.__name__}.")
        self._export(dataset, out_dir)

    @abstractmethod
    def _export(self, dataset: ZMaxDataset, out_dir: Path) -> None: ...
