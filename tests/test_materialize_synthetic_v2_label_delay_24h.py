import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.data.materialize_synthetic_v2_label_delay import (
    cohort_output_files,
    materialize_label_delay,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ID = "LABEL_DELAY_24H"
OUTPUT_FILES = cohort_output_files(SCENARIO_ID)
PREV_03_LOTS_PATH = (
    REPO_ROOT
    / "data"
    / "synthetic"
    / "v2"
    / "scenarios"
    / "PREV_03"
    / "prev_03_lots.csv"
)


def _true_mask(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "1.0", "true", "yes"})
    )


def test_label_delay_24h_preserves_prev_03_peer_cohort(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / SCENARIO_ID

    summary = materialize_label_delay(
        repo_root=REPO_ROOT,
        scenario_id=SCENARIO_ID,
        output_dir=output_dir,
    )

    assert summary.scenario_id == SCENARIO_ID
    assert summary.source_cohort_id == "PREV_03"
    assert summary.cohort_size == 1000
    assert summary.synthetic_anomaly_count == 30
    assert summary.achieved_anomaly_rate == 0.03
    assert summary.mechanism_counts == {
        "abrupt_mean_shift": 6,
        "gradual_degradation": 6,
        "variance_instability": 6,
        "sensor_fault": 6,
        "contextual_anomaly": 6,
    }
    assert summary.label_delay_hours == 24

    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).is_file()

    lots = pd.read_csv(output_dir / OUTPUT_FILES["lots"])
    source_lots = pd.read_csv(PREV_03_LOTS_PATH)

    assert lots["lot_id"].tolist() == source_lots["lot_id"].tolist()
    assert not _true_mask(lots["is_unseen_context"]).any()
    assert not _true_mask(lots["is_benign_drift"]).any()
    assert pd.to_numeric(
        lots["label_delay_hours"],
        errors="raise",
    ).eq(24).all()

    original_columns = [
        column
        for column in source_lots.columns
        if column != "label_delay_hours"
    ]
    pdt.assert_frame_equal(
        lots[original_columns].reset_index(drop=True),
        source_lots[original_columns].reset_index(drop=True),
        check_dtype=False,
    )

    assert pd.to_numeric(
        lots["source_label_delay_hours"],
        errors="raise",
    ).tolist() == pd.to_numeric(
        source_lots["label_delay_hours"],
        errors="raise",
    ).tolist()

    expected_available_at = (
        pd.to_datetime(lots["event_time"])
        + pd.Timedelta(hours=24)
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    assert (
        lots["quality_label_available_at"].astype(str).tolist()
        == expected_available_at.tolist()
    )
    assert lots["label_delay_scenario_id"].eq(
        SCENARIO_ID
    ).all()

    anomaly_lot_ids = set(
        lots.loc[
            _true_mask(lots["is_synthetic_anomaly"]),
            "lot_id",
        ].astype(str)
    )
    rca = pd.read_csv(
        output_dir / OUTPUT_FILES["rca_ground_truth"]
    )

    assert set(rca["lot_id"].astype(str)) == anomaly_lot_ids

    generated_manifest = json.loads(
        (output_dir / OUTPUT_FILES["manifest"]).read_text(
            encoding="utf-8"
        )
    )

    assert generated_manifest["source_cohort"]["id"] == "PREV_03"
    assert (
        generated_manifest["materialization"][
            "fixed_label_delay_hours"
        ]
        == 24
    )


def test_label_delay_24h_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_label_delay(
        repo_root=REPO_ROOT,
        scenario_id=SCENARIO_ID,
        output_dir=first_output_dir,
    )
    materialize_label_delay(
        repo_root=REPO_ROOT,
        scenario_id=SCENARIO_ID,
        output_dir=second_output_dir,
    )

    for filename in OUTPUT_FILES.values():
        assert (
            first_output_dir / filename
        ).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()