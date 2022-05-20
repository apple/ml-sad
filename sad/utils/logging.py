#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import logging
import logging.handlers
from datetime import datetime
from typing import List

FORMATTER_STRING = "UTC - %(asctime)s, %(name)15s, %(levelname)6s, %(message)s"


class LoggingFormatter(logging.Formatter):
    """A class that inherits from ``logging.Formatter``. Instance method ``formatTime``
    will be overwritten to produce the right format."""

    def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
        ct = datetime.utcnow()
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = f"{t}, {record.msecs:03d}"
        return s


def setup_module_level_logger(
    logger_name_list: List[str],
    level: int = logging.INFO,
):
    """Function to setup module level loggers. The format of loggers will be set
    using to ``FORMATTER_STRING``, using class ``LoggingFormatter``.

    Args:
        logger_name_list (:obj:`List[str]`): A list of module names corresponding
            top-level namespace of loggers.
        level (:obj:`int`): The level of loggers, determining the threshold of logging
            severity.
    """
    fmt_string = FORMATTER_STRING

    formatter = LoggingFormatter(fmt=fmt_string, datefmt="%Y-%m-%d,%H:%M:%S.%f")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    for logger_name in logger_name_list:
        logger = logging.getLogger(logger_name)
        logger.addHandler(console_handler)
        logger.setLevel(level=level)
