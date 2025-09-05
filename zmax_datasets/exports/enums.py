from enum import Enum, auto


class ErrorHandling(Enum):
    RAISE = auto()
    SKIP = auto()


class ExistingFileHandling(Enum):
    RAISE = "raise"
    SKIP = "skip"
    OVERWRITE = "overwrite"
    APPEND = "append"
