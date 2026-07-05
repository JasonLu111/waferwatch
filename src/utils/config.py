"""
Project-wide configuration for WaferWatch.

This file defines important folder paths used across the project.
It helps keep data ingestion, preprocessing, modeling, reporting,
and monitoring code consistent.
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Basic project information
# ---------------------------------------------------------------------

PROJECT_NAME = "WaferWatch"


# ---------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------
# __file__ means: the current Python file path.
# Path(__file__).resolve() turns it into an absolute path.
#
# Current file:
# D:/waferwatch/src/utils/config.py
#
# parents[0] -> D:/waferwatch/src/utils
# parents[1] -> D:/waferwatch/src
# parents[2] -> D:/waferwatch
#
# Therefore, parents[2] is the project root.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

# Backward-compatible aliases used by older modules.
DATA_RAW_DIR = RAW_DATA_DIR
DATA_INTERIM_DIR = INTERIM_DATA_DIR
DATA_PROCESSED_DIR = PROCESSED_DATA_DIR
DATA_SYNTHETIC_DIR = SYNTHETIC_DATA_DIR
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


# ---------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------

def ensure_directories_exist() -> None:
    """
    Create important project folders if they do not already exist.

    This prevents errors when later scripts try to save files into
    data/, reports/, or models/.
    """
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        SYNTHETIC_DATA_DIR,
        REPORTS_DIR,
        DOCS_DIR,
        MODELS_DIR,
        DASHBOARD_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------
# This block only runs when you execute:
# python src/utils/config.py
#
# It will NOT run when another file imports config.py.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ensure_directories_exist()

    print("WaferWatch configuration loaded successfully.")
    print(f"Project name: {PROJECT_NAME}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"Models directory: {MODELS_DIR}")