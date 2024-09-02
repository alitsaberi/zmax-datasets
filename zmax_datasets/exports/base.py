import logging
from abc import ABC, abstractmethod
from pathlib import Path

from zmax_datasets.datasets.base import ZMaxDataset

logger = logging.getLogger(__name__)


class ExportStrategy(ABC):
    def export(self, dataset: ZMaxDataset, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting dataset using {self.__class__.__name__}.")
        self._export(dataset, out_dir)

    @abstractmethod
    def _export(self, dataset: ZMaxDataset, out_dir: Path) -> None: ...
