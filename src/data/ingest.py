"""
Data ingestion utilities for WaferWatch.

This module is responsible for loading raw CSV files from data/raw/,
checking that they exist, and optionally saving standardized copies
to data/interim/.

At this stage, we are not cleaning the data yet.
We only load and inspect it safely.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import RAW_DATA_DIR, INTERIM_DATA_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_csv(file_name: str, data_dir: str | Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load a CSV file from a given data directory.

    Parameters
    ----------
    file_name:
        CSV file name, for example 'secom.data.csv'.
    data_dir:
        Directory where the CSV file is stored. Default is data/raw/.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.
    """

    file_path = Path(data_dir) / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}. "
            "Please place the file under data/raw/ first."
        )

    logger.info("Loading CSV file from: %s", file_path)

    df = pd.read_csv(file_path)

    logger.info(
        "Loaded CSV successfully. Rows: %s, Columns: %s",
        df.shape[0],
        df.shape[1],
    )

    return df


def save_interim_csv(
    df: pd.DataFrame,
    file_name: str,
    output_dir: str | Path = INTERIM_DATA_DIR,
) -> Path:
    """
    Save a DataFrame to data/interim/.

    Parameters
    ----------
    df:
        DataFrame to save.
    file_name:
        Output CSV file name.
    output_dir:
        Output directory. Default is data/interim/.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """

    ensure_directories_exist()

    output_path = Path(output_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.info("Saved interim CSV to: %s", output_path)

    return output_path


def _demo() -> None:
    """
    Run a small demo without needing the real dataset.

    This creates a tiny synthetic raw dataset, saves it to data/raw/,
    loads it back using load_csv(), and saves a copy to data/interim/.
    """

    ensure_directories_exist()

    demo_raw_path = RAW_DATA_DIR / "demo_sensor_data.csv"

    demo_df = pd.DataFrame(
        {
            "lot_id": ["LOT_001", "LOT_002", "LOT_003"],
            "sensor_001": [1.20, 1.35, 1.50],
            "sensor_002": [5.10, 5.30, 5.80],
            "pass_fail_label": [0, 0, 1],
        }
    )

    demo_df.to_csv(demo_raw_path, index=False)

    logger.info("Created demo raw CSV at: %s", demo_raw_path)

    loaded_df = load_csv("demo_sensor_data.csv")

    print(loaded_df)

    save_interim_csv(loaded_df, "demo_sensor_data_interim.csv")


if __name__ == "__main__":
    _demo()