from pathlib import Path

############################ Paths #############################

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging.yaml"
DATA_DIR = BASE_DIR / "data"

########################### Defaults ############################

DEFAULTS = {
    "period_length": 30,  # seconds
    "hypnogram_label": "UNKNOWN",
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
}

############################# ZMax ##############################

YASA = {
    "sampling_frequency": 100,
    "hypnogram_mapping": {
        "W": "W",
        "N1": "N1",
        "N2": "N2",
        "N3": "N3",
        "REM": "REM",
        "UNKNOWN": "Uns",
    },
}

#################################################################
