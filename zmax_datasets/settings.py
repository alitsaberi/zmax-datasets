from pathlib import Path

############################ Paths #############################

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
CONFIG_DIR = BASE_DIR / "configs"
# TODO: should be removed later when the scripts are all have a config file argument
DATASETS_CONFIG_FILE = CONFIG_DIR / "raw_datasets.yaml"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
RESOURCES_DIR = BASE_DIR / "resources"
MODELS_DIR = RESOURCES_DIR / "models"

############################ General ############################

PACKAGE_NAME = PACKAGE_DIR.name

########################### Defaults ############################

DEFAULTS = {
    "period_length": 30,  # seconds
    # TODO: change hypnogram to sleep_stage in all files for consistent naming
    "label": "UNKNOWN",
    "hypnogram_mapping": {
        0: "W",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
        -1: "UNKNOWN",
        -2: "ARTIFACT",
    },
    "hypnogram_reverse_mapping": {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4,
        "UNKNOWN": -1,
        "ARTIFACT": -2,
    },
}

########################### Logging #############################

LOGGING = {
    "log_file": LOGS_DIR / f"{PACKAGE_NAME}.log",
    "handlers": {
        "console": {
            "level": "INFO",
            "colorize": True,
            "backtrace": False,
        },
        "file": {
            "level": "DEBUG",
            "rotation": "50 MB",
            "enqueue": True,
            "backtrace": True,
        },
    },
}

############################ USleep #############################

USLEEP = {
    "sampling_frequency": 128.0,
    "data_types_file_extension": "h5",
    "hypnogram_file_extension": "ids",
}

############################# ZMax ##############################

ZMAX = {
    "data_types_file_extension": "edf",
    "sampling_frequency": 256.0,
}

############################# ZMax ##############################

YASA = {
    "sampling_frequency": 100.0,
    "hypnogram_mapping": {
        "W": "W",
        "N1": "N1",
        "N2": "N2",
        "N3": "N3",
        "REM": "R",
        "UNKNOWN": "Uns",
    },
    "hypnogram_column": "stage",
    "split_labels": {
        "train": "training",
        "test": "testing",
    },
}

############################# Artifact Detection ##############################

ARTIFACT_DETECTION = {
    "models_dir": MODELS_DIR / "eeg_usability",
    "models_info_url": "https://drive.usercontent.google.com/download?id=1f55ko0vH8BUYO9HqAQAVH4iGTu3Xdyf8&export=download&authuser=0",
    "sampling_frequency": 256.0,
    "epoch_duration": 10,
    "n_features": {
        "tsfel": 36 + 18 + 336,
        "lite": 2838,
    },
    "labels_file_extension": "csv",
}

##################################### IBI #####################################

IBI = {
    "sampling_frequency": 2.0,
    "segment_duration": 30,
}

################################# Sleep Scoring ################################

SLEEP_SCORING = {
    "filter": {
        "low_cutoff": 0.3,
        "high_cutoff": 30,
    }
}

################################################################################
