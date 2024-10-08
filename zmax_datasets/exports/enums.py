from enum import Enum, auto


class ExistingFileHandling(Enum):
    RAISE_ERROR = auto()
    OVERWRITE = auto()
    APPEND = auto()


class ErrorHandling(Enum):
    RAISE = auto()
    SKIP = auto()
