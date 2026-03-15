"""
Configuration module for the Smartphone Price Prediction Dashboard.

This module defines the global configuration settings, including file paths
for data storage, default scraping URLs, and logging directory paths.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories mapping to DVC structure
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
INTERIM_DATA_DIR = DATA_DIR / "02_interim"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"

# Logging directory
LOG_DIR = PROJECT_ROOT / "logs"

# Default configuration parameters for Data Extraction
REQUEST_TIMEOUT = 10  # Seconds to wait for server response
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def ensure_directories_exist() -> None:
    """
    Ensure all critical application directories exist on the filesystem.
    
    Time Complexity: O(D) where D is the number of directories to create.
    Space Complexity: O(1) as memory usage is minimal and independent of input data size.
    """
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        LOG_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Automatically ensure critical directories are present when config is loaded
ensure_directories_exist()
