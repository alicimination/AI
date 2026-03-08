"""Centralized logging configuration for the Reliable Multimodal Math Mentor."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "math_mentor", log_file: str = "app.log") -> logging.Logger:
    """Return a configured logger instance.

    Args:
        name: Logger name.
        log_file: Relative path to the log file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
