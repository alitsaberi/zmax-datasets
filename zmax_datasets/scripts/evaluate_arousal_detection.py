import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from slumber.processing.arousal_detection import detect_arousals
from slumber.processing.sleep_scoring import UTimeModel, score
from slumber.utils.data import Data
from utime.utils.system import find_and_set_gpus

from zmax_datasets import datasets, settings
from zmax_datasets.datasets.base import (
    Dataset,
    SleepAnnotations,
)
from zmax_datasets.utils.helpers import get_class_by_name, load_yaml_config
from zmax_datasets.utils.logger import setup_logging

logger = logging.getLogger(__name__)


MODEL_ARGS = {
    "model_dir": settings.BASE_DIR / "../utime_EEG_10",
    "n_periods": 30 * 6,
    "n_samples_per_prediction": 128,
}


def _calculate_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary"
    )
    accuracy = accuracy_score(true_labels, predicted_labels)

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def _get_data(data_file_path: Path) -> Data:
    with h5py.File(data_file_path) as h5_file:
        eeg_l = h5_file["channels"]["F7-Fpz"][:]
        eeg_r = h5_file["channels"]["F8-Fpz"][:]

        return Data(
            np.column_stack((eeg_l, eeg_r)),
            h5_file.attrs["sample_rate"],
            channel_names=["F7-Fpz", "F8-Fpz"],
        )


def _arousal_per_period(
    arousal_intervals: list[tuple[int, int]], data: Data, period_duration: int = 30
) -> np.ndarray:
    total_periods = int(np.ceil(data.duration.total_seconds() / period_duration))
    arousal_per_period = np.zeros(total_periods, dtype=int)
    samples_per_period = int(data.sample_rate * period_duration)

    # Iterate over arousal intervals and mark corresponding epochs
    for start_sample, end_sample in arousal_intervals:
        start_epoch = start_sample // samples_per_period
        end_epoch = end_sample // samples_per_period
        arousal_per_period[start_epoch : end_epoch + 1] = 1

    return arousal_per_period


def _process_dataset(
    dataset: Dataset, data_dir: Path, model: UTimeModel
) -> list[dict[str, Any]]:
    recordings = list(dataset.get_recordings(with_sleep_scoring=True))
    n_recordings = len(recordings)
    logger.info(f"Loaded {n_recordings} recordings")

    results = []

    for i, recording in enumerate(recordings):
        logger.info(f"-> Recording {i+1}/{n_recordings}: {recording}")

        data_file_path = data_dir / f"{recording}" / f"{recording}.h5"

        if not data_file_path.exists():
            logger.warning(f"Skipping {recording} because data file does not exist")
            continue

        data = _get_data(data_file_path)
        logger.info(f"Data shape: {data.shape}")

        predicted_sleep_scores = score(
            data, model, channel_groups=[[0], [1]], arg_max=False
        )
        logger.info(f"Predicted sleep scores shape: {predicted_sleep_scores.shape}")
        logger.info(f"Predicted sleep scores: {predicted_sleep_scores[:5]}")

        arousal_intervals = detect_arousals(
            predicted_sleep_scores, wake_n1_threshold=0.4
        )

        predicted_arousal_labels = _arousal_per_period(
            arousal_intervals, predicted_sleep_scores
        )

        true_arousal_labels = recording.read_annotations(
            annotation_type=SleepAnnotations.AROUSAL,
        )[: len(predicted_arousal_labels)]
        true_sleep_scoring_labels = recording.read_annotations(
            annotation_type=SleepAnnotations.SLEEP_STAGE,
        )[: len(predicted_arousal_labels)]

        deep_sleep_periods = np.where(~np.isin(true_sleep_scoring_labels, [0, 1, -1]))[
            0
        ]

        predicted_arousal_labels = predicted_arousal_labels[deep_sleep_periods]
        true_arousal_labels = true_arousal_labels[deep_sleep_periods]

        if np.all(true_arousal_labels == 0):
            logger.warning(f"Skipping {recording} because all arousal labels are 0")
            continue

        logger.info(f"Predicted arousal labels length: {len(predicted_arousal_labels)}")
        logger.info(
            f"Predicted arousal labels:"
            f" {np.unique(predicted_arousal_labels, return_counts=True)}"
        )

        logger.info(f"True arousal labels length: {len(true_arousal_labels)}")
        logger.info(
            f"True arousal labels: {np.unique(true_arousal_labels, return_counts=True)}"
        )

        metrics = _calculate_metrics(
            true_arousal_labels,
            predicted_arousal_labels,
        )
        logger.info(f"Metrics: {metrics}")
        metrics["recording"] = f"{recording}"
        metrics["dataset"] = f"{dataset}"
        results.append(metrics)

        tf.keras.backend.clear_session()

    return results


def main(config: dict[str, Any]) -> None:
    find_and_set_gpus(1)

    logger.info("Loading model...")
    model = UTimeModel(**MODEL_ARGS)

    all_results = []

    for dataset_name, dataset_config in config.items():
        dataset_class = get_class_by_name(dataset_name, datasets, Dataset)

        dataset = dataset_class(**dataset_config)
        logger.info(f"Loaded dataset: {dataset_name}")
        data_dir = settings.DATA_DIR / str.lower(dataset_name)

        if not data_dir.exists():
            logger.info(f"Data directory {data_dir} does not exist. Skipping...")
            continue

        dataset_result = _process_dataset(dataset, data_dir, model)
        logger.info(f"{len(dataset_result)} recordings processed in {dataset_name}.")
        all_results.extend(dataset_result)

    df = pd.DataFrame(all_results).set_index(["dataset", "recording"])
    # Get counts per dataset
    dataset_counts = df.groupby("dataset").size()

    # Calculate mean metrics per dataset
    dataset_metrics = df.groupby("dataset").mean()

    # Add counts as a new column
    dataset_metrics["recording_count"] = dataset_counts

    # Calculate overall macro metrics
    total_metrics = df.mean()
    total_metrics["recording_count"] = len(df)

    # Add total metrics as a new row
    dataset_metrics.loc["TOTAL"] = total_metrics

    logger.info("\nMacro metrics per dataset (with recording counts):")
    logger.info(dataset_metrics)


if __name__ == "__main__":
    log_file_name = (
        f"{Path(__file__).stem}_{datetime.now().isoformat(timespec='seconds')}"
    )
    setup_logging(file_name=log_file_name)
    config_file = settings.CONFIG_DIR / "datasets.yaml"
    config = load_yaml_config(config_file)
    main(config)
