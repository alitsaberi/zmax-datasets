from pathlib import Path

############################  Paths  ############################

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging.yaml"
DATA_DIR = BASE_DIR / "data"

############################ USleep #############################

USLEEP = {
    "sampling_frequency": 128,
}

#################################################################
