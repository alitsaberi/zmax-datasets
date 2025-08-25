from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from zmax_datasets import datasets
from zmax_datasets.datasets.base import Dataset
from zmax_datasets.utils.helpers import create_class_by_name_resolver


class DatasetConfig(BaseModel):
    name: str
    dataset: Annotated[
        type[Dataset],
        BeforeValidator(create_class_by_name_resolver(datasets, Dataset)),
    ] = Field(..., alias="dataset_type")
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def configure(self) -> "Dataset":
        return self.dataset(**self.config)
