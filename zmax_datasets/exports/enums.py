from enum import Enum, auto


class ErrorHandling(Enum):
    RAISE = auto()
    SKIP = auto()


class ExistingFileHandling(Enum):
    RAISE = auto()
    SKIP = auto()
    OVERWRITE = auto()
    APPEND = auto()
