"""Logging module for Dynamic Pricing RL.

Configures and provides a standardized, double-destination (file + console)
logger for pipeline components.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "pipeline", log_file: Optional[Path] = None
) -> logging.Logger:
    """Creates or retrieves a logger with standard console and file outputs.

    Args:
        name: Name of the logger.
        log_file: Optional Path to log file. If omitted, it will try to resolve
          default log path.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)

    # If logger is already configured, return it to avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Standard formatter: [Timestamp] [Level] [LoggerName] [Message]
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file is None:
        # Avoid direct import circularities by resolving locally if needed
        from src.config import LOG_FILE_PATH

        log_file = LOG_FILE_PATH

    try:
        # Ensure log parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(
            f"Failed to initialize file logger at {log_file} due to: {e}. "
            "Continuing with console logger only."
        )

    return logger
