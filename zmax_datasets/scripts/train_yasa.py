import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier, callback, early_stopping
from sklearn.metrics import (
    classification_report,
)
from tqdm import tqdm

from zmax_datasets import settings
from zmax_datasets.exports.yasa import summarize_dataset
from zmax_datasets.utils.helpers import load_yaml_config
from zmax_datasets.utils.logger import setup_logging
from zmax_datasets.utils.yasa import load_model

logger = logging.getLogger(__name__)

_DATA_TYPES = ["eeg", "eog", "emg", "demo"]
_TIME_FEATURE_TYPE = "time"
_CLASSIFIER_ARGS_KEY = "classifier_args"
_EVAL_METRIC = "multi_logloss"
_GENERAL_ARGUMENTS = {
    "data_file": {"type": Path, "required": True, "help": "Path to the data file"},
    "data_types": {
        "nargs": "+",
        "default": ["eeg", "eog"],
        "choices": _DATA_TYPES,
        "help": "Data types to use (can be multiple)",
    },
    "early_stopping_rounds": {
        "type": int,
        "default": 50,
        "help": "Number of early stopping rounds",
    },
    "use_pretrained": {
        "action": "store_true",
        "help": "Use pre-trained YASA model based on the chosen data types",
    },
    "ignore_unscored_samples": {
        "action": "store_true",
        "help": "Ignore samples with Uns hypnogram label",
    },
}
_CLASSIFIER_ARGUMENTS = {
    "n_jobs": {"type": int, "default": 8, "help": "Number of parallel jobs to run"},
    "class_weight": {
        "type": str,
        "default": None,
        "choices": ["balanced", "custom"],
        "help": "Class weight strategy",
    },
    "n_estimators": {
        "type": int,
        "default": 400,
        "help": "Number of boosting iterations",
    },
    "max_depth": {"type": int, "default": 5, "help": "Maximum tree depth"},
    "learning_rate": {"type": float, "default": 0.1, "help": "Boosting learning rate"},
    "num_leaves": {
        "type": int,
        "default": 90,
        "help": "Maximum tree leaves for base learners",
    },
    "colsample_bytree": {
        "type": float,
        "default": 0.5,
        "help": "Subsample ratio of columns when constructing each tree",
    },
    "importance_type": {
        "type": str,
        "default": "gain",
        "choices": ["split", "gain"],
        "help": "The type of feature importance to be filled into feature_importances_",
    },
    "boosting_type": {
        "type": str,
        "default": "gbdt",
        "choices": ["gbdt", "dart", "rf"],
        "help": "Boosting type",
    },
    "random_state": {"type": int, "default": 123, "help": "Random seed"},
    "verbose": {"type": int, "default": -1, "help": "Verbosity level"},
}


class TQDMProgressBar:
    def __init__(
        self,
        num_iterations: int,
        description: str | None = None,
        unit: str = "iteration",
    ) -> None:
        self.progress_bar = tqdm(total=num_iterations, desc=description, unit=unit)

    def __call__(self, env: callback.CallbackEnv) -> None:
        self.progress_bar.update(env.iteration - self.progress_bar.n + 1)

    def __del__(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()


def _parse_arguments() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train LGBMClassifier (YASA)")
    subparsers = parser.add_subparsers(dest="mode")

    # Config file subparser
    config_parser = subparsers.add_parser("config", help="Use a config file")
    config_parser.add_argument("config_file", type=Path, help="Path to the config file")

    # Command-line arguments subparser
    cli_parser = subparsers.add_parser("cli", help="Use command-line arguments")
    for key, value in _GENERAL_ARGUMENTS.items():
        cli_parser.add_argument(f"--{key}", **value)

    classifier_group = cli_parser.add_argument_group("Classifier Parameters")
    _add_classifier_arguments(classifier_group)

    args = parser.parse_args()

    if args.mode == "config":
        return _load_config_file(args.config_file)
    elif args.mode == "cli":
        return {
            "data_file": args.data_file,
            "data_types": _validate_data_types(args.data_types),
            "early_stopping_rounds": args.early_stopping_rounds,
            "use_pretrained": args.use_pretrained,
            "ignore_unscored_samples": args.ignore_unscored_samples,
            _CLASSIFIER_ARGS_KEY: {
                key: getattr(args, key) for key in _CLASSIFIER_ARGUMENTS
            },
        }


def _add_classifier_arguments(parser: argparse._ArgumentGroup) -> None:
    for key, value in _CLASSIFIER_ARGUMENTS.items():
        parser.add_argument(f"--{key}", **value)


def _load_config_file(config_file: Path) -> dict[str, Any]:
    config = load_yaml_config(config_file)

    if "data_file" not in config:
        raise ValueError("Data file is missssing in the config file.")

    config["data_file"] = Path(config["data_file"])
    config["data_types"] = _validate_data_types(config.get("data_types", _DATA_TYPES))
    config["early_stopping_rounds"] = config.get(
        "early_stopping_rounds", _GENERAL_ARGUMENTS["early_stopping_rounds"]["default"]
    )
    config["use_pretrained"] = config.get("use_pretrained", False)
    config["ignore_unscored_samples"] = config.get("ignore_unscored_samples", True)

    classifier_args = {k: v["default"] for k, v in _CLASSIFIER_ARGUMENTS.items()}
    classifier_args.update(config.get(_CLASSIFIER_ARGS_KEY, {}))
    config[_CLASSIFIER_ARGS_KEY] = classifier_args

    known_args = set(list(_GENERAL_ARGUMENTS.keys()) + [_CLASSIFIER_ARGS_KEY])
    unknown_args = set(config.keys()) - known_args
    if unknown_args:
        logger.warning(f"Unknown arguments in config file: {', '.join(unknown_args)}")

    return config


def _validate_data_types(data_types: list[str]) -> list[str]:
    invalid_data_types = set(data_types) - set(_DATA_TYPES)
    if invalid_data_types:
        raise ValueError(f"Invalid data types: {invalid_data_types}")
    return data_types


def main(args: dict[str, Any]) -> None:
    logger.info(f"Arguments: {args}")
    X_train, y_train, X_test, y_test = _load_data(
        args["data_file"],
        args["data_types"],
        ignore_unscored_samples=args["ignore_unscored_samples"],
    )

    if args["use_pretrained"]:
        logger.info(
            "Loading pretrained model...This will ignore any classifier arguments."
        )
        classifier = load_model(args["data_types"])
    else:
        classifier = LGBMClassifier(**args[_CLASSIFIER_ARGS_KEY])

    callbacks = [
        TQDMProgressBar(classifier.n_estimators, description="Training progress")
    ]

    if args["early_stopping_rounds"] >= 0:
        callbacks.append(
            early_stopping(
                stopping_rounds=args["early_stopping_rounds"],
                min_delta=0.0,
                verbose=True,
            )
        )

    classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=_EVAL_METRIC,
        callbacks=callbacks,
    )

    _log_scores(classifier, X_train, y_train, X_test, y_test)


def _load_data(
    data_file: Path,
    data_types: list[str],
    ignore_unscored_samples: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    logger.info(f"Loading data from {data_file}")
    df = pd.read_parquet(data_file)

    if ignore_unscored_samples:
        logger.info("Ignoring unscored samples...")
        df = df[df["stage"] != "Uns"]

    logger.info(f"Dataset summary:\n{summarize_dataset(df)}")

    feature_columns = [
        col
        for feature_type in set(data_types + [_TIME_FEATURE_TYPE])
        for col in df
        if col.startswith(f"{feature_type}_")
    ]

    logger.info(f"{len(feature_columns)} features will be used for training.")

    return (
        df.loc[settings.YASA["split_labels"]["train"], feature_columns],
        df.loc[
            settings.YASA["split_labels"]["train"], settings.YASA["hypnogram_column"]
        ],
        df.loc[settings.YASA["split_labels"]["test"], feature_columns],
        df.loc[
            settings.YASA["split_labels"]["test"], settings.YASA["hypnogram_column"]
        ],
    )


def _log_scores(
    classifier: LGBMClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)

    logger.info(f"Training Scores:\n{classification_report(y_train, y_pred_train)}")
    logger.info(f"Test Scores:\n{classification_report(y_test, y_pred_test)}")


if __name__ == "__main__":
    log_file_name = f"yasa_train_{datetime.now().isoformat(timespec='seconds')}"
    setup_logging(file_name=log_file_name)
    args = _parse_arguments()
    main(args)
