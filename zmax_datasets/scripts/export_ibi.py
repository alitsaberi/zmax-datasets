import argparse
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from zmax_datasets import settings
from zmax_datasets.datasets.base import Dataset, DataTypeMapping
from zmax_datasets.exports.ibi import IBIExportStrategy
from zmax_datasets.exports.usleep import (
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
from zmax_datasets.utils.transforms import fir_filter, resample

MEASURES = [
    "bpm",
    "ibi",
    "sdnn",
    "sdsd",
    "rmssd",
    "pnn20",
    "pnn50",
    "hr_mad",
    "sd1",
    "sd2",
    "s",
    "sd1/sd2",
    "breathingrate",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export datasets")
    parser.add_argument("output_dir", help="Output directory for exports", type=Path)
    parser.add_argument(
        "--datasets", nargs="+", help="List of datasets to export", type=str
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Channels to extract measures from",
        type=str,
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        help="Segment duration in seconds",
        default=settings.IBI["segment_duration"],
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        help="Sample rate in Hz",
        default=settings.IBI["sampling_frequency"],
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--skip-missing-data-types",
        action="store_true",
        help="Skip missing data types",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip a recording if an error occurs",
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
    channels: list[str],
) -> list[DataTypeMapping]:
    return [
        DataTypeMapping(
            channel,
            [channel],
            transforms=[
                (
                    fir_filter,
                    {"sampling_frequency": 64, "low_cutoff": 0.7, "high_cutoff": 3.5},
                ),
                (resample, {"sampling_frequency": 128, "old_sampling_frequency": 64}),
            ],
        )
        for channel in channels
    ]


def main() -> None:
    log_file = LOGS_DIR / generate_timestamped_file_name(Path(__file__).stem, "log")
    setup_logging(log_file=log_file)
    args = parse_arguments()

    logger.info(f"Arguments: {args}")

    datasets = _get_datasets(args.datasets)
    data_mappings = _get_data_mappings(args.channels)

    missing_data_type_handling = (
        ErrorHandling.SKIP if args.skip_missing_data_types else ErrorHandling.RAISE
    )
    error_handling = ErrorHandling.SKIP if args.skip_errors else ErrorHandling.RAISE
    existing_file_handling = (
        ExistingFileHandling.OVERWRITE
        if args.overwrite
        else ExistingFileHandling.RAISE_ERROR
    )

    for dataset_name, dataset in datasets.items():
        logger.info(f"Exporting dataset: {dataset_name}")

        export_strategy = IBIExportStrategy(
            data_type_mappings=data_mappings,
            sampling_frequency=args.sample_rate,
            segment_duration=args.segment_duration,
            existing_file_handling=existing_file_handling,
            missing_data_type_handling=missing_data_type_handling,
            error_handling=error_handling,
        )

        export_strategy.export(dataset, Path(args.output_dir) / dataset_name)


if __name__ == "__main__":
    main()
