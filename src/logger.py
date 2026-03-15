"""
Logging configuration for the Smartphone Price Prediction pipeline.

This module sets up Python's built-in logging to ensure standard output
visibility and rolling log file creation to prevent unbounded log expansion.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import LOG_DIR


def get_logger(module_name: str) -> logging.Logger:
    """
    Instantiate and configure a logger instance for pipeline processes.
    
    This factory function defines a standardized log format and attaches
    both a standard output stream handler and a persistent rolling file handler.
    
    Args:
        module_name (str): Name of the calling module (typically `__name__`).
        
    Returns:
        logging.Logger: A configured instance of the standard Python Logger.
        
    Time Complexity: O(1) for initialization.
    Space Complexity: O(1) relative to working memory (disk space capped by RotatingFileHandler).
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # If handlers already exist, prevent duplicate log lines
    if logger.hasHandlers():
        return logger

    # Shared formatter for clarity & debugging
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Standard output handler (System stream)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Persistent file output (max 5MB per file, keeping last 3)
    log_file_path = LOG_DIR / "pipeline.log"
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger
