import argparse
import copy
import glob
from pathlib import Path

import numpy as np
from loguru import logger

from zmax_datasets import settings
from zmax_datasets.datasets.zmax import Recording
from zmax_datasets.processing.sleep_scoring import (
    SLEEP_STAGE_CHANNEL_NAME,
    UTimeModel,
    score,
)
from zmax_datasets.sources.zmax.enums import DataTypes
from zmax_datasets.sources.zmax.utils import load_data
from zmax_datasets.transforms.filter import FIRFilter
from zmax_datasets.utils.data import Data
from zmax_datasets.utils.helpers import generate_timestamped_file_name
from zmax_datasets.utils.logger import setup_logging

EEG_CHANNELS = [data_type.name for data_type in DataTypes.get_by_category("EEG")]
FILTER = FIRFilter(
    low_cutoff=settings.SLEEP_SCORING["filter"]["low_cutoff"],
    high_cutoff=settings.SLEEP_SCORING["filter"]["high_cutoff"],
)
AGGREGATED_KEY = "aggregated"

DEFAULT_MODEL_DIR = settings.MODELS_DIR / "usleep-2024-zmax-finetune"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score sleep stages from one or multiple ZMax recordings"
    )
    parser.add_argument(
        "recording_paths",
        type=str,
        help=(
            "Path pattern(s) to ZMax recording directories. "
            "Supports glob patterns (e.g., 'data/*/recording*' or"
            " 'data/single_recording')"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Path to U-Time model directory",
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--weight-file",
        type=str,
        help="Name of specific weight file to use",
        default=None,
    )
    parser.add_argument(
        "--n-samples-per-prediction",
        type=int,
        help=(
            "Number of samples per prediction."
            " Defaults to n_samples_per_periods parameter of the model,"
            " giving 1 segmentation per period of signal. Set this to 1 to score "
            " every data point in the signal."
        ),
        default=None,
    )
    parser.add_argument(
        "--channels",
        type=str,
        choices=EEG_CHANNELS,
        nargs="+",
        help=(
            "Comma-separated list of channel names to score. "
            "Defaults to both EEG channels."
        ),
        default=EEG_CHANNELS,
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Do not apply a FIR filter to the channels",
    )
    parser.add_argument(
        "--confidences",
        action="store_true",
        help=(
            "Store confidence scores for each class instead of predicted class labels."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help=(
            "Threshold for confidence of dominant class. "
            "Only used when --confidences is False. "
            "Predictions with confidence scores below this threshold will be set to -1 "
            "(unknown)."
        ),
        default=0.0,
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help=(
            "Aggregate predictions across channels in addition to outputting "
            "predictions for each channel. When arg_max is True, this will "
            "output a majority vote across channels. In this case, if there is no "
            "majority, the prediction will be set to -1 (unknown). "
            "If arg_max is False, this will output the mean across channels."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Directory to save all predictions. If not specified, predictions will "
            "be saved in each recording directory"
        ),
        default=None,
    )
    # TODO: add --overwrite option
    # TODO: add --continue option
    # TODO: add --n-gpus option
    return parser.parse_args()


def _get_recordings(recording_paths: str) -> list[Recording]:
    return [Recording(Path(p)) for p in glob.glob(recording_paths, recursive=True)]


def _get_model(
    model_dir: Path, weight_file_name: str | None, n_samples_per_prediction: int | None
) -> UTimeModel:
    return UTimeModel(
        model_dir,
        weight_file_name=weight_file_name,
        n_samples_per_prediction=n_samples_per_prediction,
    )


def _score(
    data: Data,
    model: UTimeModel,
    channel_groups: list[list[int | str]],
    confidences: bool,
    confidence_threshold: float,
) -> Data:
    scores = score(
        data,
        model,
        channel_groups=channel_groups,
        arg_max=False,  # always return confidences
    )

    if not confidences:
        logger.info(
            f"Converting to class labels (threshold: {confidence_threshold:.2f})",
        )
        conf_values = np.max(scores.array, axis=1)
        class_labels = np.argmax(scores.array, axis=1)
        unknown_val = settings.DEFAULTS["hypnogram_reverse_mapping"]["UNKNOWN"]
        class_labels[conf_values < confidence_threshold] = unknown_val
        # Reshape to (n_samples, 1) for Data class compatibility
        scores.array = class_labels.reshape(-1, 1)
        # Update channel names for class labels
        scores.channel_names = [SLEEP_STAGE_CHANNEL_NAME]

    return scores


def _merge_scores(scores_dict: dict[str, Data], confidences: bool) -> dict[str, Data]:
    """Merge scores from multiple channels.

    For confidence values: takes mean across channels
    For class labels: uses majority voting with special handling for unknown (-1)
    - both must agree, otherwise unknown
    - if one channel is unknown (-1), uses other channel's prediction
    """

    first_scores = next(iter(scores_dict.values()))

    if len(scores_dict) == 1:
        logger.warning("Only one channel, no need to merge scores")
        scores_dict[AGGREGATED_KEY] = copy.deepcopy(first_scores)
        return scores_dict

    arrays = [scores.array for scores in scores_dict.values()]

    if confidences:
        # For confidence values (shape: n_channels x n_samples x n_classes)
        arrays = np.stack(arrays)  # stack along first dimension
        logger.info("Merging confidence values by taking mean across channels")
        merged_array = np.mean(arrays, axis=0)  # average across channels
    else:
        # For class labels (shape: n_channels x n_samples x 1)
        # Squeeze out the last dimension to get (n_channels x n_samples)
        arrays = np.stack([arr.squeeze(-1) for arr in arrays])
        logger.info("Merging class labels using majority voting")
        unknown_class = settings.DEFAULTS["hypnogram_reverse_mapping"]["UNKNOWN"]
        merged_array = np.full_like(arrays[0], unknown_class)

        # Where both channels agree
        mask_agree = arrays[0] == arrays[1]
        merged_array[mask_agree] = arrays[0][mask_agree]

        # Where one channel is unknown, use the other channel
        for i in range(2):
            mask_other_unknown = (arrays[i] != unknown_class) & (
                arrays[1 - i] == unknown_class
            )
            merged_array[mask_other_unknown] = arrays[i][mask_other_unknown]

        # Reshape back to (n_samples, 1) for Data class compatibility
        merged_array = merged_array.reshape(-1, 1)

    merged_scores = Data(
        array=merged_array,
        channel_names=[SLEEP_STAGE_CHANNEL_NAME]
        if not confidences
        else first_scores.channel_names,
        sample_rate=first_scores.sample_rate,
        timestamps=first_scores.timestamps,
    )
    scores_dict[AGGREGATED_KEY] = merged_scores
    return scores_dict


def _process_recording(
    recording: Recording,
    model: UTimeModel,
    data_types: list[DataTypes],
    filter: FIRFilter,
    args: argparse.Namespace,
) -> None:
    """Process a single recording directory."""

    logger.info("Loading data")
    data = load_data(
        recording, data_types=data_types
    )  # TODO: handle missing data types

    if not args.no_filter:
        logger.info("Applying FIR filter")
        data = filter(data)

    model.n_periods = int(
        np.floor(data.duration.total_seconds() / model.period_duration)
    )

    logger.info("Scoring sleep stages")
    scores_dict = {
        dt.name: _score(
            data,
            model,
            [[dt.name]],
            args.confidences,
            args.confidence_threshold,
        )
        for dt in data_types
    }

    if args.aggregate:
        logger.info("Aggregating scores across channels")
        scores_dict = _merge_scores(scores_dict, args.confidences)

    if args.output_path:
        output_dir = args.output_path / recording.data_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = recording.data_dir

    logger.info(f"Saving predictions to {output_dir}")
    for name, scores in scores_dict.items():
        scores.to_csv(output_dir / f"{name}.csv")


def main() -> None:
    log_file = settings.LOGS_DIR / generate_timestamped_file_name(
        Path(__file__).stem, "log"
    )
    setup_logging(log_file=log_file)
    args = _parse_arguments()

    recordings = _get_recordings(args.recording_paths)
    if not recordings:
        raise ValueError(f"No recordings found for {args.recording_paths}")

    logger.info(f"Found {len(recordings)} recording(s)")

    logger.info(f"Loading model from {args.model_dir}")
    model = _get_model(args.model_dir, args.weight_file, args.n_samples_per_prediction)

    data_types = [DataTypes[channel] for channel in args.channels]

    for i, recording in enumerate(recordings):
        logger.info(f"Processing {i+1}/{len(recordings)}: {recording}")

        try:
            _process_recording(
                recording=recording,
                model=model,
                data_types=data_types,
                filter=FILTER,
                args=args,
            )
        except Exception as e:
            logger.exception(f"Error processing {recording}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
