import pickle

import numpy as np
import requests
import tsfel
from joblib import Parallel, delayed
from loguru import logger
from scipy.signal import spectrogram

from zmax_datasets.settings import ARTIFACT_DETECTION
from zmax_datasets.utils.data import Data

MODEL_NAMES = {
    "default": "eegUsability_model_v1.0.pkl",
    "lite": "eegUsability_model_v0.7_lite.pkl",
    "binary": "eegUsability_model_v0.6_binary.pkl",
    "lite_binary": "eegUsability_model_v0.7.3_lite_binary.pkl",
}


def _get_available_models() -> dict[str, str]:
    response = requests.get(ARTIFACT_DETECTION["models_info_url"])
    response.raise_for_status()
    return response.json()


def _download_model(model_name: str) -> None:
    available_models = _get_available_models()
    if model_name not in available_models:
        raise ValueError(
            f"Model {model_name} not found in available models. "
            f"Available models: {available_models.keys()}"
        )

    model_url = available_models[model_name]
    model_path = ARTIFACT_DETECTION["models_dir"] / model_name
    response = requests.get(model_url)
    response.raise_for_status()

    with open(model_path, "wb") as f:
        f.write(response.content)


def _extract_spectrogram_features(data: np.ndarray, sample_rate: int) -> np.ndarray:
    num_epochs, num_channels, _ = data.shape

    data_reshaped = data.reshape(-1, data.shape[-1])

    _, _, Sxx = spectrogram(data_reshaped, sample_rate, axis=-1)

    # Reshape output back to (epochs, channels, freq_bins, time_bins)
    spectrogram_features = Sxx.reshape(num_epochs, num_channels, *Sxx.shape[1:])

    # Transpose to match expected dimensions
    spectrogram_features = np.transpose(spectrogram_features, (0, 1, 3, 2))
    logger.debug(f"Spectrogram features shape: {spectrogram_features.shape}")

    return spectrogram_features.astype(np.float32)


def _extract_tsfel_features_per_channel(
    data: np.ndarray, sample_rate: int
) -> np.ndarray:
    # Cache the feature configurations
    global CGF_STATISTICAL, CGF_TEMPORAL, CGF_SPECTRAL
    if "CGF_STATISTICAL" not in globals():
        CGF_STATISTICAL = tsfel.get_features_by_domain("statistical")
        CGF_TEMPORAL = tsfel.get_features_by_domain("temporal")
        CGF_SPECTRAL = tsfel.get_features_by_domain("spectral")

    # Extract all features at once instead of separately
    all_features = {}
    all_features.update(
        tsfel.time_series_features_extractor(
            CGF_STATISTICAL, data, fs=sample_rate, verbose=0
        )
    )
    all_features.update(
        tsfel.time_series_features_extractor(
            CGF_TEMPORAL, data, fs=sample_rate, verbose=0
        )
    )
    all_features.update(
        tsfel.time_series_features_extractor(
            CGF_SPECTRAL, data, fs=sample_rate, verbose=0
        )
    )

    return np.array(list(all_features.values()), dtype=np.float32).T


def _extract_tsfel_features(data: np.ndarray, sample_rate: int) -> np.ndarray:
    num_epochs, num_channels, _ = data.shape

    # Reshape data to process all epochs and channels at once
    reshaped_data = data.reshape(-1, data.shape[-1])

    # Process all epochs and channels in parallel
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_extract_tsfel_features_per_channel)(epoch_channel_data, sample_rate)
        for epoch_channel_data in reshaped_data
    )

    # Convert results to numpy array and reshape back to original dimensions
    tsfel_features = np.array(results).reshape(num_epochs, num_channels, -1)
    logger.debug(f"tsfel features shape: {tsfel_features.shape}")

    return tsfel_features


def _create_samples(
    spectogram_features: np.ndarray, statistical_features: np.ndarray
) -> np.ndarray:
    logger.info("Creating samples for usability detection...")
    # Preparing Sectrogram features:
    spec_feats_l = spectogram_features[:, [0, 2], :, :]  # EEG_L and movement
    spec_feats_r = spectogram_features[:, [1, 2], :, :]  # EEG_R and movement
    spec_feats_l = np.hstack(
        (spec_feats_l[:, 0, :, :], spec_feats_l[:, 1, :, :])
    )  # Stacking EEG_L and movement features side-by-side
    temp = [
        spec_feats_l[:, i, :] for i in range(spec_feats_l.shape[1])
    ]  # Flattening features
    spec_feats_l = np.hstack(temp)  # Stacking flattened features
    spec_feats_r = np.hstack((spec_feats_r[:, 0, :, :], spec_feats_r[:, 1, :, :]))
    temp = [spec_feats_r[:, i, :] for i in range(spec_feats_r.shape[1])]
    spec_feats_r = np.hstack(temp)
    # Preparing Satistical features:
    stat_feats_usability_l = statistical_features[:, [0, 2], :]  # EEG_L and movement
    stat_feats_usability_r = statistical_features[:, [1, 2], :]  # EEG_R and movement
    stat_feats_usability_l = np.hstack(
        (stat_feats_usability_l[:, 0, :], stat_feats_usability_l[:, 1, :])
    )  # Stacking EEG_L and movement features side-by-side
    stat_feats_usability_r = np.hstack(
        (stat_feats_usability_r[:, 0, :], stat_feats_usability_r[:, 1, :])
    )
    # Stacking Sectrogram and Satistical features side-by-side
    x_test_l = np.hstack((spec_feats_l, stat_feats_usability_l))
    x_test_r = np.hstack((spec_feats_r, stat_feats_usability_r))

    return x_test_l, x_test_r


def _create_lite_samples(spectrogram_features: np.ndarray) -> np.ndarray:
    logger.info("Creating samples for lite usability detection...")

    if spectrogram_features.shape[1] != 3:
        raise ValueError(
            "Expected 3 channels (EEG_L, EEG_R, movement) in spectrogram_features."
        )

    # Flatten EEG and movement features for each epoch
    eeg_left = spectrogram_features[:, 0, :, :].reshape(
        spectrogram_features.shape[0], -1
    )
    eeg_right = spectrogram_features[:, 1, :, :].reshape(
        spectrogram_features.shape[0], -1
    )
    movement = spectrogram_features[:, 2, :, :].reshape(
        spectrogram_features.shape[0], -1
    )

    # Concatenate EEG and movement features for each channel
    features_left = np.concatenate([eeg_left, movement], axis=1)
    features_right = np.concatenate([eeg_right, movement], axis=1)

    return features_left, features_right


def load_model(version: str = "default") -> object:
    model_name = MODEL_NAMES.get(version)

    if model_name is None:
        raise ValueError(
            f"Model {version} not found in available models. "
            f"Available models: {MODEL_NAMES.keys()}"
        )

    model_path = ARTIFACT_DETECTION["models_dir"] / model_name

    if not model_path.exists():
        logger.info(
            f"Model {model_name} not found in "
            f"{ARTIFACT_DETECTION['models_dir']}. Downloading..."
        )
        _download_model(model_name)
        logger.info(f"Model {model_name} downloaded to {model_path}")

    with open(model_path, "rb") as f:
        logger.info(f"Loading model {model_name} from {model_path}")
        data = pickle.load(f)
        return data["model"]


def get_usability_scores(
    data: Data,
    model: object,
    eeg_left_label: str,
    eeg_right_label: str,
    movement_label: str,
) -> tuple[Data, int, int]:
    """
    Get usability scores for a given data.

    Args:
        data: Data object with channels EEG_L in uV, EEG_R in uV, and movement
        model: Model object

    Returns:
        usability_scores: Data object of usability scores with shape (n_epochs, 2)
            columns are left and right channels
        samples_to_keep: number of samples to keep
        epoch_length: length of each epoch
    """

    if data.sample_rate != ARTIFACT_DETECTION["sampling_frequency"]:
        raise ValueError(
            f"Data must have a sample rate of"
            f" {ARTIFACT_DETECTION['sampling_frequency']}"
        )

    expected_order = [eeg_left_label, eeg_right_label, movement_label]

    if set(expected_order) - set(data.channel_names):
        raise ValueError(f"Data must have the following channels: {expected_order}")

    data = data[:, expected_order]

    epoch_length = int(ARTIFACT_DETECTION["epoch_duration"] * data.sample_rate)
    n_epochs = data.length // epoch_length
    samples_to_keep = n_epochs * epoch_length
    logger.info(
        f"Number of samples: {data.length},"
        f" Number of epochs: {n_epochs},"
        f" Samples to keep: {samples_to_keep}"
    )

    if n_epochs == 0:
        raise ValueError(
            "No epochs found in the data",
            epoch_length=epoch_length,
            data_length=data.length,
        )

    if samples_to_keep < data.length:
        logger.info(
            f"Dropping {data.length - samples_to_keep}"
            f" samples from the end of the data."
        )
        data = data[:samples_to_keep]

    array = data.array.reshape(n_epochs, epoch_length, data.n_channels).transpose(
        0, 2, 1
    )

    spectrogram_features = _extract_spectrogram_features(array, data.sample_rate)

    logger.debug(f"Model features: {model.num_feature()}")
    if model.num_feature() == ARTIFACT_DETECTION["n_features"]["lite"]:
        logger.info(f"Using lite set of features: {model.num_feature()}")
        samples_left, samples_right = _create_lite_samples(spectrogram_features)
    else:
        logger.info(f"Using full set of features: {model.num_feature()}")
        tsfel_features = _extract_tsfel_features(array, data.sample_rate)
        samples_left, samples_right = _create_samples(
            spectrogram_features, tsfel_features
        )
        del tsfel_features

    logger.debug(f"Samples left shape: {samples_left.shape}")
    logger.debug(f"Samples right shape: {samples_right.shape}")

    del spectrogram_features

    predictions_left = model.predict(samples_left)
    predictions_right = model.predict(samples_right)

    usability_scores = np.column_stack(
        (np.argmax(predictions_left, axis=1), np.argmax(predictions_right, axis=1))
    )
    logger.debug(f"Usability scores shape: {usability_scores.shape}")

    return (
        Data(
            array=usability_scores,
            sample_rate=1.0 / ARTIFACT_DETECTION["epoch_duration"],
            channel_names=ARTIFACT_DETECTION["channel_names"],
            timestamp_offset=np.mean(data.timestamps[:epoch_length]),
        ),
        samples_to_keep,
        epoch_length,
    )
