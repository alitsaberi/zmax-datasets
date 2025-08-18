import logging.config
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from zmax_datasets import settings


class HandlerType(Enum):
    CONSOLE = "console"
    FILE = "file"


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    log_file: Path = settings.LOGGING["log_file"],
    config: dict[str, Any] = settings.LOGGING,
    intercept_standard_logging: bool = False,
) -> None:
    logger.remove()

    for handler_name, handler_config in config["handlers"].items():
        handler_type = HandlerType(handler_name)

        match handler_type:
            case HandlerType.CONSOLE:
                sink = sys.stdout
            case HandlerType.FILE:
                sink = log_file

        logger.add(sink, **handler_config)

    # Intercept standard library logging
    # This way, all logs from the root logger and other libraries using standard
    # logging will be properly formatted and handled by Loguru
    if intercept_standard_logging:
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
