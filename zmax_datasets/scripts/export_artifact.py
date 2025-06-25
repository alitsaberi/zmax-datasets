import argparse
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from zmax_datasets import settings
from zmax_datasets.datasets.base import Dataset, DataTypeMapping
from zmax_datasets.exports.artifact import (
    ArtifactExportStrategy,
    ErrorHandling,
    ExistingFileHandling,
)
from zmax_datasets.scripts.export import DatasetConfig
from zmax_datasets.settings import LOGS_DIR
from zmax_datasets.utils.helpers import (
    generate_timestamped_file_name,
    load_yaml_config,
)
from zmax_datasets.utils.logger import setup_logging
from zmax_datasets.utils.transforms import l2_normalize


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export artifact data")
    parser.add_argument("output_dir", help="Output directory for exports", type=Path)
    parser.add_argument(
        "--datasets", nargs="+", help="List of datasets to export", type=str
    )
    parser.add_argument(
        "--eeg-left",
        help="EEG left channel to extract",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--eeg-right",
        help="EEG right channel to extract",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--accel",
        nargs=3,
        help="Accelerometer channels to extract",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--skip-errors", action="store_true", help="Skip a recording if an error occurs"
    )
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
    eeg_left_channel: str, eeg_right_channel: str, accel_channels: list[str]
) -> list[DataTypeMapping]:
    return [
        DataTypeMapping(eeg_left_channel, [eeg_left_channel]),
        DataTypeMapping(eeg_right_channel, [eeg_right_channel]),
        DataTypeMapping("movement", accel_channels, transforms=[l2_normalize]),
    ]


def main() -> None:
    log_file = LOGS_DIR / generate_timestamped_file_name(Path(__file__).stem, "log")
    setup_logging(log_file=log_file)
    args = parse_arguments()

    logger.info(f"Arguments: {args}")

    datasets = _get_datasets(args.datasets)
    data_mappings = _get_data_mappings(args.eeg_left, args.eeg_right, args.accel)

    error_handling = ErrorHandling.SKIP if args.skip_errors else ErrorHandling.RAISE
    existing_file_handling = (
        ExistingFileHandling.OVERWRITE
        if args.overwrite
        else ExistingFileHandling.RAISE_ERROR
    )

    for dataset_name, dataset in datasets.items():
        logger.info(f"Exporting dataset: {dataset_name}")

        export_strategy = ArtifactExportStrategy(
            data_type_mappings=data_mappings,
            existing_file_handling=existing_file_handling,
            error_handling=error_handling,
        )

        export_strategy.export(dataset, Path(args.output_dir) / dataset_name)


if __name__ == "__main__":
    main()
