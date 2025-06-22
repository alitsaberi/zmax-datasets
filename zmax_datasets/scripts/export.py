import argparse
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter

from zmax_datasets import datasets, settings
from zmax_datasets.datasets.base import DATA_TYPES, Dataset, DataTypeMapping
from zmax_datasets.exports.usleep import (
    ErrorHandling,
    ExistingFileHandling,
    USleepExportStrategy,
)
from zmax_datasets.settings import LOGS_DIR
from zmax_datasets.utils.helpers import (
    create_class_by_name_resolver,
    generate_timestamped_file_name,
    load_yaml_config,
)
from zmax_datasets.utils.logger import setup_logging


class DatasetConfig(BaseModel):
    name: str
    dataset: Annotated[
        type[Dataset],
        BeforeValidator(create_class_by_name_resolver(datasets, Dataset)),
    ] = Field(..., alias="class_name")
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def configure(self) -> "Dataset":
        return self.dataset(**self.config)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export datasets")
    parser.add_argument("output_dir", help="Output directory for exports", type=Path)
    parser.add_argument(
        "--datasets", nargs="+", help="List of datasets to export", type=str
    )
    parser.add_argument("--channels", nargs="+", help="Channels to extract", type=str)
    parser.add_argument(
        "--rename-channels", nargs="+", help="New names for the channels", type=str
    )
    parser.add_argument(
        "--resample", action="store_true", help="Whether to resample data"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument("--skip-errors", action="store_true", help="Skip errors")
    return parser.parse_args()


def _get_datasets(datasets_to_export: list[str]) -> dict[str, Dataset]:
    datasets_config = TypeAdapter(list[DatasetConfig]).validate_python(
        load_yaml_config(settings.DATASETS_CONFIG_FILE)
    )
    logger.info(f"Datasets: {datasets_config}")
    available_datasets = [dataset.name for dataset in datasets_config]
    if invalid_datasets := set(datasets_to_export) - set(available_datasets):
        raise ValueError(
            f"Invalid dataset name: {invalid_datasets}. "
            f"Available datasets: {available_datasets}"
        )
    return {
        dataset.name: dataset.configure()
        for dataset in datasets_config
        if dataset.name in datasets_to_export
    }


def _get_data_mappings(
    channels: list[str], rename_channels: list[str]
) -> list[DataTypeMapping]:
    if invalid_channels := set(channels) - set(DATA_TYPES):
        raise ValueError(
            f"Invalid channel name: {invalid_channels}. "
            f"Available channels: {DATA_TYPES}"
        )

    rename_channels = rename_channels or channels

    if len(channels) != len(rename_channels):
        raise ValueError(
            f"Number of channels and rename channels must be the same. "
            f"Got {len(channels)} channels and {len(rename_channels)} rename channels."
        )

    return [
        DataTypeMapping(rename_channel, [channel])
        for channel, rename_channel in zip(channels, rename_channels, strict=True)
    ]


def main() -> None:
    log_file = LOGS_DIR / generate_timestamped_file_name(Path(__file__).stem, "log")
    setup_logging(log_file=log_file)
    args = parse_arguments()

    logger.info(f"Arguments: {args}")

    datasets = _get_datasets(args.datasets)
    data_mappings = _get_data_mappings(args.channels, args.rename_channels)

    for dataset_name, dataset in datasets.items():
        logger.info(f"Exporting dataset: {dataset_name}")

        export_strategy = USleepExportStrategy(
            data_type_mappigns=data_mappings,
            sampling_frequency=settings.USLEEP["sampling_frequency"]
            if args.resample
            else settings.ZMAX["sampling_frequency"],
            existing_file_handling=ExistingFileHandling.OVERWRITE
            if args.overwrite
            else ExistingFileHandling.APPEND,
            error_handling=ErrorHandling.SKIP
            if args.skip_errors
            else ErrorHandling.RAISE,
        )

        export_strategy.export(dataset, Path(args.output_dir) / dataset_name)


if __name__ == "__main__":
    main()
