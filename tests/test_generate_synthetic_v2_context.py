from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.generate_synthetic_v2_context import (
    apply_abrupt_mean_shift,
    build_baseline_tables,
    normalize_quality_label,
    validate_generated_tables,
)
from src.data.validate_synthetic_v2_config import load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "synthetic_data_v2.json"


def sample_secom_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lot_id": [
                "SOURCE_LOT_001",
                "SOURCE_LOT_002",
                "SOURCE_LOT_003",
                "SOURCE_LOT_004",
            ],
            "sensor_001": [0.1, 0.2, 0.3, 0.4],
            "sensor_002": [1.0, 1.1, 1.2, 1.3],
            "pass_fail_label": [-1, 1, -1, 1],
        }
    )


def test_normalize_quality_label_converts_secom_labels() -> None:
    labels = pd.Series([-1, 1, -1, 1])

    normalized = normalize_quality_label(labels)

    assert normalized.tolist() == [0, 1, 0, 1]


def test_baseline_generation_is_deterministic() -> None:
    config = load_config(CONFIG_PATH)
    source = sample_secom_frame()

    first_tables = build_baseline_tables(source, config)
    second_tables = build_baseline_tables(source, config)

    for table_id in first_tables:
        assert_frame_equal(
            first_tables[table_id],
            second_tables[table_id],
        )


def test_baseline_tables_match_the_synthetic_v2_contract() -> None:
    config = load_config(CONFIG_PATH)
    tables = build_baseline_tables(sample_secom_frame(), config)

    validate_generated_tables(tables, config)

    lots = tables["lots"]

    assert lots["lot_id"].is_unique
    assert lots["source_lot_id"].tolist() == [
        "SOURCE_LOT_001",
        "SOURCE_LOT_002",
        "SOURCE_LOT_003",
        "SOURCE_LOT_004",
    ]
    assert set(lots["quality_label"].unique()) == {0, 1}
    assert lots["is_synthetic_anomaly"].eq(0).all()
    assert lots["anomaly_mechanism"].eq("none").all()
    assert lots["root_cause_id"].eq("none").all()
    assert (lots["label_available_at"] > lots["event_time"]).all()
    assert set(tables) == {
        "lots",
        "tool_events",
        "maintenance",
        "process_changes",
        "rca_ground_truth",
    }
    
def test_abrupt_mean_shift_creates_grounded_evidence() -> None:
    config = load_config(CONFIG_PATH)

    source = pd.DataFrame(
        {
            "lot_id": [
                f"SOURCE_LOT_{index:03d}"
                for index in range(1, 101)
            ],
            "sensor_001": list(range(100)),
            "sensor_002": [index * 2 for index in range(100)],
            "sensor_003": [index * 3 for index in range(100)],
            "pass_fail_label": [-1] * 100,
        }
    )

    baseline_tables = build_baseline_tables(source, config)
    injected_tables = apply_abrupt_mean_shift(
        baseline_tables,
        config,
    )

    validate_generated_tables(injected_tables, config)

    injected_lots = injected_tables["lots"].loc[
        injected_tables["lots"]["anomaly_mechanism"]
        == "abrupt_mean_shift"
    ]

    assert len(injected_lots) == 3
    assert injected_lots["is_synthetic_anomaly"].eq(1).all()
    assert injected_lots["root_cause_id"].eq(
        "RC_PRESSURE_INSTABILITY"
    ).all()
    assert injected_lots["synthetic_evidence_id"].ne("").all()
    assert injected_lots["injected_sensor_columns"].ne("").all()
    assert injected_lots["abrupt_shift_sigma"].gt(0).all()

    abrupt_events = injected_tables["tool_events"].loc[
        injected_tables["tool_events"]["event_id"] == "EVT_ABRUPT_001"
    ]
    assert len(abrupt_events) == 1
    assert abrupt_events.iloc[0]["alarm_code"] == "ALARM_PRESSURE_SHIFT"

    abrupt_cases = injected_tables["rca_ground_truth"]
    assert len(abrupt_cases) == len(injected_lots)
    assert abrupt_cases["evidence_ids"].str.contains(
        "EVID_ALARM_ABRUPT_001"
    ).all()