from pathlib import Path

############################ Paths #############################

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
CONFIG_DIR = BASE_DIR / "configs"
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging.yaml"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

############################ General ############################

PACKAGE_NAME = PACKAGE_DIR.name
LOG_FILE_EXTENSION = ".log.jsonl"

########################### Defaults ############################

DEFAULTS = {
    "period_length": 30,  # seconds
    # TODO: change hypnogram to sleep_stage in all files for consistent naming
    "label": "UNKNOWN",
    "hynogram_mapping": {
        0: "W",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
        -1: "UNKNOWN",
    },
}

############################ USleep #############################

USLEEP = {
    "sampling_frequency": 128,
    "data_types_file_extension": "h5",
    "hypnogram_file_extension": "ids",
}

############################# ZMax ##############################

ZMAX = {
    "data_types_file_extension": "edf",
    "sampling_frequency": 256,
}

############################# ZMax ##############################

YASA = {
    "sampling_frequency": 100,
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

#################################################################
