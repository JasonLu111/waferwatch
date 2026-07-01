"""
Logging utilities for WaferWatch.

This module provides a reusable logger for data pipelines, model training,
monitoring scripts, API services, and dashboard components.
"""

import logging
import sys


def get_logger(name: str = "waferwatch") -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name:
        Name of the logger. Using different names can help identify
        which module produced each log message.

    Returns
    -------
    logging.Logger
        A configured logger object.
    """

    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if get_logger() is called multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger()

    logger.info("WaferWatch logger loaded successfully.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")