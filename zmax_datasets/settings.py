from pathlib import Path

############################ Paths #############################

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging.yaml"
DATA_DIR = BASE_DIR / "data"

########################### Defaults ############################

DEFAULTS = {
    "period_length": 30,  # seconds
}

############################ USleep #############################

USLEEP = {
    "sampling_frequency": 128,
    "default_hypnogram_label": "UNKNOWN",
    "data_types_file_extension": "h5",
    "hypnogram_file_extension": "ids",
}

#################################################################
