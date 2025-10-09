import argparse
from pathlib import Path

from loguru import logger

from zmax_datasets.datasets.base import Dataset
from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.exports.usleep import CATALOG_FILE_NAME, USleepExportStrategy
from zmax_datasets.exports.utils import SleepAnnotations
from zmax_datasets.pipeline.configs import PipelineConfig
from zmax_datasets.scripts.export import DatasetConfig
from zmax_datasets.settings import LOGS_DIR
from zmax_datasets.utils.helpers import (
    generate_timestamped_file_name,
    load_yaml_config,
)
from zmax_datasets.utils.logger import setup_logging


def parse_arguments():
    """Parse command line arguments for pipeline processing"""
    parser = argparse.ArgumentParser(
        description="Process datasets using configurable pipelines"
    )

    # Required arguments
    parser.add_argument(
        "output_dir", help="Output directory for processed data", type=Path
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset-config",
        type=Path,
        help="Dataset configuration file (YAML). If not provided, uses default.",
        required=True,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=(
            "List of datasets to process. "
            "If not provided, all datasets will be processed."
        ),
        type=str,
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        help="Pipeline configuration file (YAML)",
    )
    parser.add_argument(
        "--input-data-types",
        nargs="+",
        help="Data types to read from recording",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--rename-input-data-types",
        nargs="+",
        help=(
            "Rename input data types after reading them."
            " Must have same length as --input-data-types."
            " If not provided, input data types will not be renamed."
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-data-types",
        nargs="+",
        help="Data types to write to output files",
        type=str,
        default=[],
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        help=(
            "Sample rate for resampling (Hz). "
            "If provided, applies resampling after pipeline."
        ),
    )
    parser.add_argument(
        "--annotation",
        type=SleepAnnotations,
        help="Annotation type to export",
        choices=list(SleepAnnotations),
        default=None,
    )
    parser.add_argument(
        "--existing-file-handling",
        type=ExistingFileHandling,
        help="Existing file handling",
        choices=list(ExistingFileHandling),
        default=ExistingFileHandling.RAISE,
    )
    parser.add_argument(
        "--skip-data-type-on-error",
        action="store_true",
        help="Skip data type if an error occurs",
    )
    parser.add_argument(
        "--skip-annotation-on-error",
        action="store_true",
        help="Skip annotation if an error occurs",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Skip recording if processing fails",
    )
    parser.add_argument(
        "--only-with-sleep-scoring",
        action="store_true",
        help="Only process recordings with sleep scoring",
    )
    parser.add_argument(
        "--catalog-file-name",
        type=str,
        help="Catalog file name",
        default=CATALOG_FILE_NAME,
    )

    return parser.parse_args()


def _load_datasets(
    config_file: Path, datasets_to_process: list[str] | None
) -> dict[str, Dataset]:
    """Load and configure datasets"""

    # Load dataset configurations
    datasets_config_data = load_yaml_config(config_file)
    datasets_config = [DatasetConfig(**config) for config in datasets_config_data]

    logger.info(f"Available datasets: {[ds.name for ds in datasets_config]}")
    available_datasets = [dataset.name for dataset in datasets_config]

    datasets_to_process = datasets_to_process or available_datasets

    # Validate requested datasets
    if invalid_datasets := set(datasets_to_process) - set(available_datasets):
        raise ValueError(
            f"Invalid dataset names: {invalid_datasets}. "
            f"Available datasets: {available_datasets}"
        )

    # Configure and return datasets
    return {
        dataset.name: dataset.configure()
        for dataset in datasets_config
        if dataset.name in datasets_to_process
    }


def _load_pipeline_config(pipeline_file: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file"""
    logger.info(f"Loading pipeline configuration from: {pipeline_file}")

    if not pipeline_file.exists():
        raise FileNotFoundError(
            f"Pipeline configuration file not found: {pipeline_file}"
        )

    pipeline_data = load_yaml_config(pipeline_file)
    pipeline_config = PipelineConfig.model_validate(pipeline_data)

    logger.info(f"Loaded pipeline: '{pipeline_config.name}'")
    logger.info(f"Pipeline steps: {len(pipeline_config.steps)}")

    return pipeline_config


def _create_export_strategy(
    pipeline_config: PipelineConfig,
    rename_mapping: dict[str, str] | None,
    args: argparse.Namespace,
) -> USleepExportStrategy:
    # Configure error handling
    data_type_error_handling = (
        ErrorHandling.SKIP if args.skip_data_type_on_error else ErrorHandling.RAISE
    )
    annotation_error_handling = (
        ErrorHandling.SKIP if args.skip_annotation_on_error else ErrorHandling.RAISE
    )
    error_handling = ErrorHandling.SKIP if args.skip_on_error else ErrorHandling.RAISE

    export_strategy = USleepExportStrategy(
        input_data_types=args.input_data_types,
        output_data_types=args.output_data_types,
        rename_mapping=rename_mapping,
        pipeline_config=pipeline_config,
        sample_rate=args.sample_rate,
        annotation_type=args.annotation,
        existing_file_handling=args.existing_file_handling,
        data_type_error_handling=data_type_error_handling,
        annotation_error_handling=annotation_error_handling,
        error_handling=error_handling,
        with_sleep_scoring=args.only_with_sleep_scoring,
        catalog_file_name=args.catalog_file_name,
    )

    return export_strategy


def _create_rename_mapping(
    input_data_types: list[str], rename_input_data_types: list[str] | None
) -> dict[str, str] | None:
    if rename_input_data_types is None:
        return None

    if len(input_data_types) != len(rename_input_data_types):
        raise ValueError(
            f"Number of input data types and rename input data types must be the same. "
            f"Got {len(input_data_types)} input data types and"
            f" {len(rename_input_data_types)} rename input data types."
        )

    return dict(zip(input_data_types, rename_input_data_types, strict=True))


def print_processing_summary(
    args,
    datasets: dict[str, Dataset],
    pipeline_config: PipelineConfig,
    rename_mapping: dict[str, str] | None,
):
    """Print a summary of what will be processed"""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Pipeline: {pipeline_config.name}") if pipeline_config else "None"

    if pipeline_config:
        print(f"Description: {pipeline_config.description}")
        print(f"Steps: {len(pipeline_config.steps)}")

        for i, step in enumerate(pipeline_config.steps, 1):
            print(f"  {i}. {step.input_data_types} → {step.output_data_types}")
            if step.transforms:
                print(f"     Transforms: {len(step.transforms)}")

    print(f"\nInput data types: {args.input_data_types}")
    if rename_mapping:
        print("Rename mapping:")
        for original, renamed in rename_mapping.items():
            print(f"  {original} → {renamed}")

    print(f"Output data types: {args.output_data_types}")

    if args.annotation:
        print(f"Annotations: {args.annotation}")

    print(f"\nDatasets to process ({len(datasets)}):")
    for dataset_name, dataset in datasets.items():
        print(f"  - {dataset_name}: {dataset.n_recordings} recordings")

    print(f"\nOutput directory: {args.output_dir}")
    print("=" * 60)


def main() -> None:
    """Main processing function"""

    args = parse_arguments()

    log_file = LOGS_DIR / generate_timestamped_file_name(Path(__file__).stem, "log")
    setup_logging(log_file=log_file)

    logger.info(f"Starting pipeline processing with arguments: {args}")

    try:
        datasets = _load_datasets(args.dataset_config, args.datasets)
        logger.info(f"Loaded {len(datasets)} datasets")

        pipeline_config = (
            _load_pipeline_config(args.pipeline) if args.pipeline else None
        )

        rename_mapping = _create_rename_mapping(
            args.input_data_types, args.rename_input_data_types
        )

        print_processing_summary(args, datasets, pipeline_config, rename_mapping)

        # Create export strategy
        export_strategy = _create_export_strategy(pipeline_config, rename_mapping, args)

        # Process each dataset
        print("\n🚀 Starting processing...")
        for dataset_name, dataset in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            print(f"\n📊 Processing dataset: {dataset_name}")

            output_dir = args.output_dir / dataset_name
            export_strategy.export(dataset, output_dir)

            print(f"✅ Completed dataset: {dataset_name}")

        print("\n🎉 All processing completed successfully!")
        print(f"📁 Output saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"\n❌ Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
