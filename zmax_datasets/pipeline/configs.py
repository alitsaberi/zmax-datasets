from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field

from zmax_datasets import transforms
from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.helpers import create_class_by_name_resolver


class TransformConfig(BaseModel):
    transform: Annotated[
        type[Transform],
        BeforeValidator(create_class_by_name_resolver(transforms, Transform)),
    ]
    config: dict[str, Any] = Field(default_factory=dict)


class PipelineStepConfig(BaseModel):
    input_data_types: list[str] = Field(
        ..., description="Which data types to use as input", min_length=1
    )
    output_data_types: list[str] = Field(
        ..., description="Name for the output data", min_length=1
    )
    transforms: list[TransformConfig] = Field(
        default_factory=list, description="Transforms to apply"
    )


class PipelineConfig(BaseModel):
    """Simple pipeline configuration"""

    name: str = Field(..., description="Pipeline name", min_length=1)
    description: str = Field(default="", description="Pipeline description")
    steps: list[PipelineStepConfig] = Field(
        ..., description="List of processing steps", min_length=1
    )
