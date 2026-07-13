from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.generate_synthetic_v2_context import (
    apply_abrupt_mean_shift,
    apply_gradual_degradation,
    apply_recipe_product_mix_drift,
    apply_sensor_faults,
    apply_variance_instability,
    build_baseline_tables,
    normalize_quality_label,
    validate_generated_tables,
    validate_recipe_product_mix_drift,
    validate_sensor_faults,
    validate_variance_instability,
    apply_contextual_anomaly,
    validate_contextual_anomaly,
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
            "sensor_002": [
                index * 2
                for index in range(100)
            ],
            "sensor_003": [
                index * 3
                for index in range(100)
            ],
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


def test_gradual_degradation_creates_delayed_maintenance_evidence() -> None:
    config = load_config(CONFIG_PATH)

    source = pd.DataFrame(
        {
            "lot_id": [
                f"SOURCE_LOT_{index:04d}"
                for index in range(1, 1001)
            ],
            "sensor_001": list(range(1000)),
            "sensor_002": [index * 2 for index in range(1000)],
            "sensor_003": [index * 3 for index in range(1000)],
            "pass_fail_label": [-1] * 1000,
        }
    )

    baseline_tables = build_baseline_tables(source, config)
    abrupt_tables = apply_abrupt_mean_shift(
        baseline_tables,
        config,
    )
    injected_tables = apply_gradual_degradation(
        abrupt_tables,
        config,
    )

    validate_generated_tables(injected_tables, config)

    lots = injected_tables["lots"]
    abrupt_lots = lots.loc[
        lots["anomaly_mechanism"] == "abrupt_mean_shift"
    ]
    gradual_lots = lots.loc[
        lots["anomaly_mechanism"] == "gradual_degradation"
    ]

    assert len(gradual_lots) == 30
    assert gradual_lots["is_synthetic_anomaly"].eq(1).all()
    assert gradual_lots["root_cause_id"].eq(
        "RC_COMPONENT_WEAR"
    ).all()
    assert gradual_lots["synthetic_evidence_id"].ne("").all()
    assert gradual_lots["maintenance_evidence_id"].eq(
        "EVID_MAINT_DELAY_GRADUAL_001"
    ).all()
    assert gradual_lots["degradation_final_sigma"].gt(0).all()

    progress = gradual_lots.sort_values(
        "event_time"
    )["degradation_progress"]
    assert progress.is_monotonic_increasing
    assert progress.iloc[-1] == 1.0

    abrupt_contexts = set(
        zip(abrupt_lots["tool_id"], abrupt_lots["chamber_id"])
    )
    gradual_contexts = set(
        zip(gradual_lots["tool_id"], gradual_lots["chamber_id"])
    )
    assert abrupt_contexts.isdisjoint(gradual_contexts)

    delayed_maintenance = injected_tables["maintenance"].loc[
        injected_tables["maintenance"]["maintenance_id"]
        == "MAINT_GRADUAL_001"
    ]
    assert len(delayed_maintenance) == 1
    assert int(delayed_maintenance.iloc[0]["delay_days"]) in {2, 3, 4, 5}
    assert delayed_maintenance.iloc[0]["evidence_id"] == (
        "EVID_MAINT_DELAY_GRADUAL_001"
    )

    gradual_cases = injected_tables["rca_ground_truth"].loc[
        injected_tables["rca_ground_truth"]["root_cause_id"]
        == "RC_COMPONENT_WEAR"
    ]
    assert len(gradual_cases) == len(gradual_lots)
    assert gradual_cases["evidence_ids"].str.contains(
        "EVID_MAINT_DELAY_GRADUAL_001"
    ).all()


def test_variance_instability_preserves_mean_and_adds_evidence() -> None:
    config = load_config(CONFIG_PATH)

    source = pd.DataFrame(
        {
            "lot_id": [
                f"SOURCE_LOT_{index:04d}"
                for index in range(1, 1501)
            ],
            "sensor_001": list(range(1500)),
            "sensor_002": [index * 2 for index in range(1500)],
            "sensor_003": [index * 3 for index in range(1500)],
            "pass_fail_label": [-1] * 1500,
        }
    )

    baseline_tables = build_baseline_tables(source, config)
    abrupt_tables = apply_abrupt_mean_shift(
        baseline_tables,
        config,
    )
    gradual_tables = apply_gradual_degradation(
        abrupt_tables,
        config,
    )
    injected_tables = apply_variance_instability(
        gradual_tables,
        config,
    )

    validate_generated_tables(injected_tables, config)
    validate_variance_instability(injected_tables)
    
    lots = injected_tables["lots"]
    variance_lots = lots.loc[
        lots["anomaly_mechanism"] == "variance_instability"
    ]

    assert len(variance_lots) == 45
    assert variance_lots["is_synthetic_anomaly"].eq(1).all()
    assert variance_lots["root_cause_id"].eq(
        "RC_PROCESS_VARIABILITY"
    ).all()
    assert variance_lots["synthetic_evidence_id"].ne("").all()
    assert variance_lots["variance_final_multiplier"].gt(1.0).all()
    assert variance_lots["variance_mean_delta"].abs().lt(1e-9).all()

    multiplier_progress = variance_lots.sort_values(
        "event_time"
    )["variance_multiplier_progress"]
    assert multiplier_progress.is_monotonic_increasing
    assert multiplier_progress.iloc[-1] > 1.0

    prior_contexts = set(
        zip(
            lots.loc[
                lots["anomaly_mechanism"].isin(
                    {"abrupt_mean_shift", "gradual_degradation"}
                ),
                "tool_id",
            ],
            lots.loc[
                lots["anomaly_mechanism"].isin(
                    {"abrupt_mean_shift", "gradual_degradation"}
                ),
                "chamber_id",
            ],
        )
    )
    variance_contexts = set(
        zip(variance_lots["tool_id"], variance_lots["chamber_id"])
    )
    assert prior_contexts.isdisjoint(variance_contexts)

    variance_events = injected_tables["tool_events"].loc[
        injected_tables["tool_events"]["event_id"]
        == "EVT_VARIANCE_001"
    ]
    assert len(variance_events) == 1
    assert variance_events.iloc[0]["alarm_code"] == (
        "ALARM_PROCESS_VARIABILITY"
    )

    variance_cases = injected_tables["rca_ground_truth"].loc[
        injected_tables["rca_ground_truth"]["root_cause_id"]
        == "RC_PROCESS_VARIABILITY"
    ]
    assert len(variance_cases) == len(variance_lots)
    assert variance_cases["evidence_ids"].str.contains(
        "EVID_VARIANCE_EVENT_001"
    ).all()



def test_sensor_faults_cover_all_fault_types_with_evidence() -> None:
    config = load_config(CONFIG_PATH)

    source = pd.DataFrame(
        {
            "lot_id": [
                f"SOURCE_LOT_{index:04d}"
                for index in range(1, 1601)
            ],
            "sensor_001": list(range(1600)),
            "sensor_002": [index * 2 for index in range(1600)],
            "sensor_003": [index * 3 for index in range(1600)],
            "sensor_004": [index * 4 for index in range(1600)],
            "pass_fail_label": [-1] * 1600,
        }
    )

    baseline_tables = build_baseline_tables(source, config)
    abrupt_tables = apply_abrupt_mean_shift(
        baseline_tables,
        config,
    )
    gradual_tables = apply_gradual_degradation(
        abrupt_tables,
        config,
    )
    variance_tables = apply_variance_instability(
        gradual_tables,
        config,
    )
    injected_tables = apply_sensor_faults(
        variance_tables,
        config,
    )

    validate_generated_tables(injected_tables, config)
    validate_variance_instability(injected_tables)
    validate_sensor_faults(injected_tables)

    lots = injected_tables["lots"]
    sensor_fault_lots = lots.loc[
        lots["anomaly_mechanism"] == "sensor_fault"
    ]

    assert len(sensor_fault_lots) == 48
    assert set(sensor_fault_lots["sensor_fault_type"]) == {
        "stuck_at",
        "dropout",
        "missing_burst",
        "calibration_bias",
    }
    assert sensor_fault_lots["root_cause_id"].eq(
        "RC_SENSOR_HARDWARE"
    ).all()
    assert sensor_fault_lots["synthetic_evidence_id"].ne("").all()

    sensor_fault_events = injected_tables["tool_events"].loc[
        injected_tables["tool_events"]["event_type"]
        == "sensor_fault_alarm"
    ]
    assert len(sensor_fault_events) == 4

    sensor_fault_cases = injected_tables["rca_ground_truth"].loc[
        injected_tables["rca_ground_truth"]["root_cause_id"]
        == "RC_SENSOR_HARDWARE"
    ]
    assert len(sensor_fault_cases) == len(sensor_fault_lots)

    missing_burst_lots = sensor_fault_lots.loc[
        sensor_fault_lots["sensor_fault_type"] == "missing_burst"
    ]
    missing_burst_columns = set(
        ";".join(
            missing_burst_lots["injected_sensor_columns"].tolist()
        ).split(";")
    )
    assert any(
        missing_burst_lots[column].isna().all()
        for column in missing_burst_columns
        if column
    )

    stuck_at_lots = sensor_fault_lots.loc[
        sensor_fault_lots["sensor_fault_type"] == "stuck_at"
    ]
    stuck_at_columns = set(
        ";".join(
            stuck_at_lots["injected_sensor_columns"].tolist()
        ).split(";")
    )
    assert any(
        stuck_at_lots[column].dropna().nunique() == 1
        for column in stuck_at_columns
        if column
    )


def test_recipe_product_mix_drift_is_benign_and_keeps_sensors_unchanged() -> None:
    config = load_config(CONFIG_PATH)

    source = pd.DataFrame(
        {
            "lot_id": [
                f"SOURCE_LOT_{index:04d}"
                for index in range(1, 1601)
            ],
            "sensor_001": list(range(1600)),
            "sensor_002": [index * 2 for index in range(1600)],
            "sensor_003": [index * 3 for index in range(1600)],
            "sensor_004": [index * 4 for index in range(1600)],
            "pass_fail_label": [-1] * 1600,
        }
    )

    baseline_tables = build_baseline_tables(source, config)
    abrupt_tables = apply_abrupt_mean_shift(
        baseline_tables,
        config,
    )
    gradual_tables = apply_gradual_degradation(
        abrupt_tables,
        config,
    )
    variance_tables = apply_variance_instability(
        gradual_tables,
        config,
    )
    sensor_fault_tables = apply_sensor_faults(
        variance_tables,
        config,
    )

    sensor_columns = [
        column
        for column in sensor_fault_tables["lots"].columns
        if str(column).startswith("sensor_")
    ]
    sensors_before_drift = sensor_fault_tables["lots"][
        sensor_columns
    ].copy()

    injected_tables = apply_recipe_product_mix_drift(
        sensor_fault_tables,
        config,
    )

    validate_generated_tables(injected_tables, config)
    validate_variance_instability(injected_tables)
    validate_sensor_faults(injected_tables)
    validate_recipe_product_mix_drift(injected_tables, config)

    lots = injected_tables["lots"]
    drift_lots = lots.loc[lots["is_benign_drift"] == 1]

    assert len(drift_lots) == 72
    assert set(drift_lots["benign_drift_type"]) == {
        "recipe_mix_change",
        "product_mix_change",
        "tool_reassignment",
    }
    assert drift_lots["is_synthetic_anomaly"].eq(0).all()
    assert drift_lots["anomaly_mechanism"].eq("none").all()
    assert drift_lots["root_cause_id"].eq("none").all()
    assert drift_lots["benign_drift_evidence_id"].ne("").all()

    assert_frame_equal(
        sensors_before_drift,
        lots[sensor_columns],
        check_dtype=True,
    )

    drift_changes = injected_tables["process_changes"].loc[
        injected_tables["process_changes"]["change_id"].str.startswith(
            "CHANGE_BENIGN_",
            na=False,
        )
    ]
    assert len(drift_changes) == 3
    assert drift_changes["is_benign_drift"].eq(True).all()

    benign_cases = injected_tables["rca_ground_truth"].loc[
        injected_tables["rca_ground_truth"]["root_cause_id"]
        == "RC_BENIGN_MIX_CHANGE"
    ]
    assert len(benign_cases) == 3
    assert benign_cases["supports_abstention"].eq(True).all()

def test_contextual_anomaly_requires_recipe_context() -> None:
    config = load_config(CONFIG_PATH)

    source = pd.DataFrame(
        {
            "lot_id": [
                f"SOURCE_LOT_{index:04d}"
                for index in range(1, 1568)
            ],
            "sensor_001": list(range(1567)),
            "sensor_002": [
                index * 2
                for index in range(1567)
            ],
            "sensor_003": [
                index * 3
                for index in range(1567)
            ],
            "pass_fail_label": [-1] * 1567,
        }
    )

    baseline_tables = build_baseline_tables(source, config)
    abrupt_tables = apply_abrupt_mean_shift(
        baseline_tables,
        config,
    )
    gradual_tables = apply_gradual_degradation(
        abrupt_tables,
        config,
    )
    variance_tables = apply_variance_instability(
        gradual_tables,
        config,
    )
    sensor_fault_tables = apply_sensor_faults(
        variance_tables,
        config,
    )
    benign_drift_tables = apply_recipe_product_mix_drift(
        sensor_fault_tables,
        config,
    )
    contextual_tables = apply_contextual_anomaly(
        benign_drift_tables,
        config,
    )

    validate_generated_tables(contextual_tables, config)
    validate_variance_instability(contextual_tables)
    validate_sensor_faults(contextual_tables)
    validate_recipe_product_mix_drift(
        tables=contextual_tables,
        config=config,
    )
    validate_contextual_anomaly(contextual_tables, config)

    lots = contextual_tables["lots"]
    expected_count = int(
        len(source)
        * config["anomaly_mechanisms"][5]["parameters"][
            "injection_rate"
        ]
    )

    anomaly_lots = lots.loc[
        lots["anomaly_mechanism"].eq("contextual_anomaly")
    ].copy()
    control_lots = lots.loc[
        lots["is_contextual_control"].eq(1)
    ].copy()

    assert len(anomaly_lots) == expected_count
    assert len(control_lots) == expected_count
    assert anomaly_lots["is_synthetic_anomaly"].eq(1).all()
    assert control_lots["is_synthetic_anomaly"].eq(0).all()
    assert control_lots["anomaly_mechanism"].eq("none").all()
    assert anomaly_lots["root_cause_id"].eq(
        "RC_RECIPE_CONTEXT_MISMATCH"
    ).all()
    assert control_lots["recipe_id"].eq("RCP_A").all()
    assert anomaly_lots["recipe_id"].eq("RCP_B").all()

    paired_lots = anomaly_lots.merge(
        control_lots,
        on="contextual_pair_id",
        suffixes=("_anomaly", "_control"),
    )
    assert len(paired_lots) == expected_count

    sensor_columns = anomaly_lots.iloc[0][
        "contextual_sensor_columns"
    ].split("|")

    for column in sensor_columns:
        assert (
            paired_lots[f"{column}_anomaly"]
            == paired_lots[f"{column}_control"]
        ).all()

    earlier_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in lots.loc[
            lots["anomaly_mechanism"].isin(
                {
                    "abrupt_mean_shift",
                    "gradual_degradation",
                    "variance_instability",
                    "sensor_fault",
                }
            ),
            ["tool_id", "chamber_id"],
        ].drop_duplicates().itertuples(index=False)
    }
    contextual_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in anomaly_lots[
            ["tool_id", "chamber_id"]
        ].drop_duplicates().itertuples(index=False)
    }

    assert len(contextual_contexts) == 1
    assert not earlier_contexts.intersection(contextual_contexts)

    contextual_events = contextual_tables["tool_events"].loc[
        contextual_tables["tool_events"]["event_id"].eq(
            "EVT_CONTEXTUAL_001"
        )
    ]
    assert len(contextual_events) == 1
    assert contextual_events["alarm_code"].eq(
        "ALARM_RECIPE_CONTEXT_MISMATCH"
    ).all()
    assert contextual_events["evidence_id"].eq(
        "EVID_CONTEXT_EVENT_001"
    ).all()

    contextual_cases = contextual_tables["rca_ground_truth"].loc[
        contextual_tables["rca_ground_truth"]["root_cause_id"].eq(
            "RC_RECIPE_CONTEXT_MISMATCH"
        )
    ]
    assert len(contextual_cases) == expected_count
    assert set(contextual_cases["lot_id"]) == set(
        anomaly_lots["lot_id"]
    )
    assert contextual_cases["evidence_ids"].str.contains(
        "EVID_CONTEXT_EVENT_001",
        regex=False,
    ).all()

