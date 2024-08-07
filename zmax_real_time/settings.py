from pathlib import Path

############################  Paths  ############################

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging.yaml"
