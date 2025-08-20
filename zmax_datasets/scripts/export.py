import argparse
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter

from zmax_datasets import datasets, settings
from zmax_datasets.datasets.base import Dataset
from zmax_datasets.exports.usleep import (
    ErrorHandling,
    ExistingFileHandling,
    USleepExportStrategy,
)
from zmax_datasets.exports.utils import DataTypeMapping, SleepAnnotations
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
        "--datasets",
        nargs="+",
        help=(
            "List of datasets to export."
            " If not provided, all datasets will be exported."
        ),
        type=str,
    )
    parser.add_argument("--channels", nargs="+", help="Channels to extract", type=str)
    parser.add_argument(
        "--rename-channels", nargs="+", help="New names for the channels", type=str
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        help=(
            "Sample rate for the exported data (Hz)."
            " If provided, all channels will be resampled to this rate and"
            " set as an attribute in the exported file."
            " If not provided, the sample rate will be inferred from the dataset"
            " for each channel and set as an attribute for the channel."
        ),
    )
    parser.add_argument(
        "--annotation",
        type=str,
        help="Annotation to export",
        choices=[annotation.name for annotation in SleepAnnotations],
    )
    parser.add_argument(
        "--skip-missing-data-types",
        action="store_true",
        help="Skip missing data types",
    )
    parser.add_argument(
        "--skip-missing-annotations",
        action="store_true",
        help="Skip missing annotations. Only used if --annotation is provided.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip a recording if an error occurs",
    )
    return parser.parse_args()


def _get_datasets(datasets_to_export: list[str] | None) -> dict[str, Dataset]:
    datasets_config = TypeAdapter(list[DatasetConfig]).validate_python(
        load_yaml_config(settings.DATASETS_CONFIG_FILE)
    )
    logger.info(f"Available datasets: {datasets_config}")
    available_datasets = [dataset.name for dataset in datasets_config]

    datasets_to_export = datasets_to_export or available_datasets

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
    channels: list[str] | None, rename_channels: list[str] | None
) -> list[DataTypeMapping]:
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
    logger.info(f"Datasets to export: {datasets}")

    data_mappings = _get_data_mappings(args.channels, args.rename_channels)

    data_type_error_handling = (
        ErrorHandling.SKIP if args.skip_missing_data_types else ErrorHandling.RAISE
    )
    annotation_error_handling = (
        ErrorHandling.SKIP if args.skip_missing_annotations else ErrorHandling.RAISE
    )
    error_handling = ErrorHandling.SKIP if args.skip_errors else ErrorHandling.RAISE
    existing_file_handling = (
        ExistingFileHandling.OVERWRITE if args.overwrite else ExistingFileHandling.RAISE
    )

    for dataset_name, dataset in datasets.items():
        logger.info(f"Exporting dataset: {dataset_name}")

        export_strategy = USleepExportStrategy(
            data_type_mappings=data_mappings,
            sample_rate=args.sample_rate,
            annotation_type=SleepAnnotations[args.annotation],
            existing_file_handling=existing_file_handling,
            data_type_error_handling=data_type_error_handling,
            annotation_error_handling=annotation_error_handling,
            error_handling=error_handling,
        )

        export_strategy.export(dataset, Path(args.output_dir) / dataset_name)


if __name__ == "__main__":
    main()
