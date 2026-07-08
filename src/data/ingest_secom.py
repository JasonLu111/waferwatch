from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/secom")
SECOM_DATA_PATH = RAW_DIR / "secom.data"
SECOM_LABELS_PATH = RAW_DIR / "secom_labels.data"
SECOM_NAMES_PATH = RAW_DIR / "secom.names"


def check_required_files() -> None:
    required_files = [
        SECOM_DATA_PATH,
        SECOM_LABELS_PATH,
        SECOM_NAMES_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing required SECOM raw files:\n"
            f"{missing_text}\n\n"
            "Please place secom.data, secom_labels.data, and secom.names "
            "under data/raw/secom/."
        )


def load_sensor_data() -> pd.DataFrame:
    sensor_df = pd.read_csv(
        SECOM_DATA_PATH,
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan", "?"],
        engine="python",
    )

    sensor_df.columns = [
        f"sensor_{index:03d}" for index in range(sensor_df.shape[1])
    ]

    return sensor_df


def load_labels() -> pd.DataFrame:
    labels_raw = pd.read_csv(
        SECOM_LABELS_PATH,
        sep=r"\s+",
        header=None,
        names=["pass_fail_label", "date", "time"],
        quotechar='"',
        engine="python",
    )

    timestamp_text = (
        labels_raw["date"].astype(str).str.replace('"', "", regex=False)
        + " "
        + labels_raw["time"].astype(str).str.replace('"', "", regex=False)
    )

    labels_df = pd.DataFrame()
    labels_df["pass_fail_label"] = labels_raw["pass_fail_label"].astype(int)
    labels_df["timestamp"] = pd.to_datetime(
        timestamp_text,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    return labels_df


def main() -> None:
    check_required_files()

    sensor_df = load_sensor_data()
    labels_df = load_labels()

    if len(sensor_df) != len(labels_df):
        raise ValueError(
            "SECOM sensor rows and label rows do not match: "
            f"{len(sensor_df)} sensor rows vs {len(labels_df)} label rows."
        )

    print("SECOM raw data loaded")
    print(f"Sensor data shape: {sensor_df.shape}")
    print(f"Label data shape: {labels_df.shape}")
    print(f"Missing sensor values: {int(sensor_df.isna().sum().sum())}")
    print(f"Missing timestamps: {int(labels_df['timestamp'].isna().sum())}")


if __name__ == "__main__":
    main()