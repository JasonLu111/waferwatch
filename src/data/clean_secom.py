from pathlib import Path

import pandas as pd

from src.data.ingest_secom import check_required_files, load_labels, load_sensor_data


PROCESSED_DIR = Path("data/processed")
CLEANED_DATA_PATH = PROCESSED_DIR / "secom_cleaned.csv"


def clean_secom_data(sensor_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    cleaned_sensor_df = sensor_df.copy()

    # 1. Remove duplicated sensor rows if any.
    duplicated_rows = int(cleaned_sensor_df.duplicated().sum())
    if duplicated_rows > 0:
        cleaned_sensor_df = cleaned_sensor_df.drop_duplicates().reset_index(drop=True)
        labels_df = labels_df.loc[cleaned_sensor_df.index].reset_index(drop=True)

    # 2. Remove high-missingness features.
    missing_rates = cleaned_sensor_df.isna().mean()
    high_missing_columns = missing_rates[missing_rates >= 0.50].index.tolist()
    cleaned_sensor_df = cleaned_sensor_df.drop(columns=high_missing_columns)

    # 3. Median-impute remaining missing values.
    medians = cleaned_sensor_df.median(numeric_only=True)
    cleaned_sensor_df = cleaned_sensor_df.fillna(medians)

    # 4. Remove constant features after imputation.
    nunique = cleaned_sensor_df.nunique(dropna=False)
    constant_columns = nunique[nunique <= 1].index.tolist()
    cleaned_sensor_df = cleaned_sensor_df.drop(columns=constant_columns)

    # 5. Build lot-level cleaned table.
    output_df = pd.DataFrame()
    output_df["lot_id"] = [f"SECOM_{index:04d}" for index in range(len(labels_df))]
    output_df["timestamp"] = labels_df["timestamp"]
    output_df["pass_fail_label"] = labels_df["pass_fail_label"]

    output_df = pd.concat([output_df, cleaned_sensor_df.reset_index(drop=True)], axis=1)

    return output_df


def main() -> None:
    check_required_files()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sensor_df = load_sensor_data()
    labels_df = load_labels()

    cleaned_df = clean_secom_data(sensor_df, labels_df)
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"Cleaned dataset saved to {CLEANED_DATA_PATH}")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Remaining missing values: {int(cleaned_df.isna().sum().sum())}")


if __name__ == "__main__":
    main()