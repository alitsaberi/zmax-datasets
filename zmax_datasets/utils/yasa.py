import logging
from pathlib import Path

import joblib
import yasa
from lightgbm import LGBMClassifier
from packaging import version

logger = logging.getLogger(__name__)


def load_model(feature_types: list[str]) -> LGBMClassifier:
    classifiers_dir = Path(yasa.__file__).parent / "classifiers"
    classifier_name = f"clf_{'+'.join(sorted(feature_types))}"
    all_matching_model_files = list(classifiers_dir.glob(f"{classifier_name}_*.joblib"))

    if not all_matching_model_files:
        raise ValueError(
            f"No pre-trained model available for feature types: {feature_types}"
        )

    latest_model = max(
        all_matching_model_files, key=lambda f: version.parse(f.stem.split("_")[-1])
    )

    logger.info(f"Using pre-trained YASA classifier: {latest_model}")
    return joblib.load(latest_model)
