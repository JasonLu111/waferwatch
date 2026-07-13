from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.validate_synthetic_v2_config import (
    SyntheticV2ConfigError,
    load_config,
    validate_config,
)


LABEL_CANDIDATES = (
    "quality_label",
    "label",
    "pass_fail",
    "pass_fail_label",
    "target",
    "failure_label",
    "class",
    "status",
)

BASELINE_RCA_COLUMNS = [
    "case_id",
    "lot_id",
    "root_cause_id",
    "suspected_cause",
    "recommended_action",
    "outcome",
    "evidence_ids",
    "supports_abstention",
    "top3_acceptable_causes",
]


def infer_label_column(frame: pd.DataFrame) -> str:
    columns_by_lower_name = {
        str(column).strip().lower(): str(column)
        for column in frame.columns
    }

    for candidate in LABEL_CANDIDATES:
        if candidate in columns_by_lower_name:
            return columns_by_lower_name[candidate]

    raise ValueError(
        "Could not infer the SECOM quality-label column. "
        f"Expected one of: {', '.join(LABEL_CANDIDATES)}. "
        f"Available columns: {', '.join(map(str, frame.columns[:20]))}"
    )


def normalize_quality_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        unique_values = set(numeric.astype(int).unique())

        if unique_values.issubset({-1, 1}):
            return (numeric.astype(int) == 1).astype("int64")

        if unique_values.issubset({0, 1}):
            return numeric.astype("int64")

    normalized_text = series.astype(str).str.strip().str.lower()

    pass_values = {"pass", "passed", "good", "normal", "ok", "0"}
    fail_values = {"fail", "failed", "bad", "abnormal", "defect", "1"}

    output = pd.Series(index=series.index, dtype="int64")
    output.loc[normalized_text.isin(pass_values)] = 0
    output.loc[normalized_text.isin(fail_values)] = 1

    if output.isna().any():
        unknown_values = sorted(
            normalized_text.loc[output.isna()].unique().tolist()
        )
        raise ValueError(
            "Could not normalize quality labels. Unknown values: "
            f"{unknown_values}"
        )

    return output.astype("int64")


def build_baseline_tables(
    base_frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    if base_frame.empty:
        raise ValueError("The SECOM input dataset must contain at least one row.")

    label_column = infer_label_column(base_frame)
    sensor_frame = base_frame.drop(columns=[label_column]).reset_index(drop=True)

    metadata_columns = {
        "lot_id",
        "source_row_id",
        "event_time",
        "tool_id",
        "chamber_id",
        "recipe_id",
        "product_family",
        "shift",
        "quality_label",
        "label_delay_hours",
        "label_available_at",
        "is_synthetic_anomaly",
        "anomaly_mechanism",
        "root_cause_id",
        "is_unseen_context",
        "split",
    }

    if "lot_id" in sensor_frame.columns:
        if "source_lot_id" in sensor_frame.columns:
            raise ValueError(
                "The SECOM input contains both lot_id and source_lot_id, "
                "so source identity cannot be renamed safely."
            )

        sensor_frame = sensor_frame.rename(
            columns={"lot_id": "source_lot_id"}
        )

    conflicting_columns = metadata_columns.intersection(sensor_frame.columns)

    if conflicting_columns:
        raise ValueError(
            "The SECOM input already contains reserved Synthetic Data V2 "
            f"columns: {sorted(conflicting_columns)}"
        )

    random_seed = config["random_seed"]
    context = config["context"]
    experiment_grid = config["experiment_grid"]

    tools = list(context["tools"])
    recipes = list(context["recipes"])
    product_families = list(context["product_families"])
    shifts = list(context["shifts"])
    chambers_per_tool = int(context["chambers_per_tool"])
    label_delays = list(experiment_grid["label_delay_hours"])

    rng = np.random.default_rng(random_seed)
    row_count = len(sensor_frame)

    event_time = pd.Timestamp("2026-01-01 06:00:00") + pd.to_timedelta(
        np.arange(row_count) * 4,
        unit="h",
    )

    assigned_tools = rng.choice(tools, size=row_count)
    chamber_numbers = rng.integers(
        low=1,
        high=chambers_per_tool + 1,
        size=row_count,
    )
    assigned_chambers = [
        f"{tool}_CH_{number:02d}"
        for tool, number in zip(assigned_tools, chamber_numbers)
    ]

    assigned_recipes = rng.choice(recipes, size=row_count)
    assigned_products = rng.choice(product_families, size=row_count)
    assigned_shifts = rng.choice(shifts, size=row_count)
    assigned_delays = rng.choice(label_delays, size=row_count)

    unseen_context = experiment_grid["unseen_context"]
    holdout_tools = set(unseen_context["holdout_tools"])
    holdout_chambers = set(unseen_context["holdout_chambers"])

    is_unseen_context = np.array(
        [
            int(tool in holdout_tools or chamber in holdout_chambers)
            for tool, chamber in zip(assigned_tools, assigned_chambers)
        ],
        dtype="int64",
    )

    standard_split = rng.choice(
        ["train", "validation", "test"],
        size=row_count,
        p=[0.70, 0.15, 0.15],
    )
    split = np.where(
        is_unseen_context == 1,
        "unseen_context",
        standard_split,
    )

    quality_label = normalize_quality_label(base_frame[label_column]).reset_index(
        drop=True
    )

    lot_metadata = pd.DataFrame(
        {
            "lot_id": [f"SYNV2_{index:05d}" for index in range(1, row_count + 1)],
            "source_row_id": np.arange(1, row_count + 1),
            "event_time": event_time,
            "tool_id": assigned_tools,
            "chamber_id": assigned_chambers,
            "recipe_id": assigned_recipes,
            "product_family": assigned_products,
            "shift": assigned_shifts,
            "quality_label": quality_label,
            "label_delay_hours": assigned_delays,
            "label_available_at": event_time
            + pd.to_timedelta(assigned_delays, unit="h"),
            "is_synthetic_anomaly": 0,
            "anomaly_mechanism": "none",
            "root_cause_id": "none",
            "is_unseen_context": is_unseen_context,
            "split": split,
        }
    )

    lots = pd.concat(
        [lot_metadata.reset_index(drop=True), sensor_frame],
        axis=1,
    )

    unique_contexts = (
        lots[["tool_id", "chamber_id"]]
        .drop_duplicates()
        .sort_values(["tool_id", "chamber_id"])
        .reset_index(drop=True)
    )

    tool_events = unique_contexts.copy()
    tool_events.insert(
        0,
        "event_id",
        [
            f"EVT_BASE_{index:03d}"
            for index in range(1, len(tool_events) + 1)
        ],
    )
    tool_events["alarm_code"] = "STATUS_ROUTINE"
    tool_events["event_type"] = "routine_status"
    tool_events["start_time"] = [
        event_time[0] + pd.Timedelta(days=index)
        for index in range(len(tool_events))
    ]
    tool_events["end_time"] = tool_events["start_time"] + pd.Timedelta(
        minutes=15
    )
    tool_events["severity"] = "low"
    tool_events["related_lot_id"] = ""
    tool_events["evidence_id"] = [
        f"EVID_BASE_EVENT_{index:03d}"
        for index in range(1, len(tool_events) + 1)
    ]

    maintenance = unique_contexts.copy()
    maintenance.insert(
        0,
        "maintenance_id",
        [
            f"MAINT_BASE_{index:03d}"
            for index in range(1, len(maintenance) + 1)
        ],
    )
    maintenance["scheduled_at"] = [
        event_time[0] + pd.Timedelta(days=7 + index)
        for index in range(len(maintenance))
    ]
    maintenance["performed_at"] = maintenance["scheduled_at"]
    maintenance["delay_days"] = 0
    maintenance["maintenance_type"] = "preventive"
    maintenance["component"] = "baseline_inspection"
    maintenance["replacement_performed"] = False
    maintenance["related_lot_id"] = ""
    maintenance["evidence_id"] = [
        f"EVID_BASE_MAINT_{index:03d}"
        for index in range(1, len(maintenance) + 1)
    ]

    change_count = min(12, row_count)
    change_positions = np.unique(
        np.linspace(0, row_count - 1, num=change_count, dtype=int)
    )

    process_change_records: list[dict[str, Any]] = []

    for record_index, lot_position in enumerate(change_positions, start=1):
        lot = lots.iloc[lot_position]
        old_recipe = str(lot["recipe_id"])
        new_recipe = recipes[
            (recipes.index(old_recipe) + 1) % len(recipes)
        ]

        process_change_records.append(
            {
                "change_id": f"CHANGE_BASE_{record_index:03d}",
                "change_type": "baseline_recipe_assignment",
                "tool_id": lot["tool_id"],
                "chamber_id": lot["chamber_id"],
                "old_value": old_recipe,
                "new_value": new_recipe,
                "changed_at": lot["event_time"] - pd.Timedelta(hours=1),
                "related_lot_id": lot["lot_id"],
                "is_benign_drift": True,
                "evidence_id": f"EVID_BASE_CHANGE_{record_index:03d}",
            }
        )

    process_changes = pd.DataFrame(process_change_records)

    rca_ground_truth = pd.DataFrame(columns=BASELINE_RCA_COLUMNS)

    return {
        "lots": lots,
        "tool_events": tool_events,
        "maintenance": maintenance,
        "process_changes": process_changes,
        "rca_ground_truth": rca_ground_truth,
    }
def apply_abrupt_mean_shift(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    updated_tables = {
        table_id: table.copy()
        for table_id, table in tables.items()
    }

    mechanism = next(
        item
        for item in config["anomaly_mechanisms"]
        if item["id"] == "abrupt_mean_shift"
    )
    parameters = mechanism["parameters"]

    lots = updated_tables["lots"]
    tool_events = updated_tables["tool_events"]
    rca_ground_truth = updated_tables["rca_ground_truth"]

    injection_rate = float(parameters["injection_rate"])
    sensor_fraction_min, sensor_fraction_max = parameters[
        "affected_sensor_fraction"
    ]
    magnitude_min, magnitude_max = parameters["magnitude_sigma"]
    alarm_code = str(parameters["alarm_code"])
    alarm_lead_hours = float(parameters["alarm_lead_hours"])

    if not 0 < injection_rate < 1:
        raise ValueError(
            "abrupt_mean_shift injection_rate must be between 0 and 1."
        )

    injection_lot_count = max(
        1,
        int(round(len(lots) * injection_rate)),
    )

    numeric_sensor_columns = [
        column
        for column in lots.columns
        if str(column).startswith("sensor_")
        and pd.api.types.is_numeric_dtype(lots[column])
    ]

    sensor_standard_deviations = lots[numeric_sensor_columns].std(
        axis=0,
        ddof=0,
    )
    eligible_sensor_columns = [
        column
        for column in numeric_sensor_columns
        if np.isfinite(sensor_standard_deviations[column])
        and sensor_standard_deviations[column] > 0
    ]

    if not eligible_sensor_columns:
        raise ValueError(
            "No non-constant numeric sensor columns are available for "
            "abrupt_mean_shift injection."
        )

    context_sizes = (
        lots.groupby(["tool_id", "chamber_id"], sort=True)
        .size()
        .reset_index(name="lot_count")
    )
    eligible_contexts = context_sizes.loc[
        context_sizes["lot_count"] >= injection_lot_count
    ].reset_index(drop=True)

    if eligible_contexts.empty:
        raise ValueError(
            "No tool/chamber context contains enough lots for the "
            "configured abrupt_mean_shift injection rate."
        )

    rng = np.random.default_rng(config["random_seed"] + 101)

    selected_context = eligible_contexts.iloc[
        rng.integers(low=0, high=len(eligible_contexts))
    ]
    selected_tool = str(selected_context["tool_id"])
    selected_chamber = str(selected_context["chamber_id"])

    episode = (
        lots.loc[
            (lots["tool_id"] == selected_tool)
            & (lots["chamber_id"] == selected_chamber)
        ]
        .sort_values("event_time")
        .head(injection_lot_count)
        .copy()
    )

    sensor_fraction = rng.uniform(
        low=float(sensor_fraction_min),
        high=float(sensor_fraction_max),
    )
    affected_sensor_count = max(
        1,
        int(round(len(eligible_sensor_columns) * sensor_fraction)),
    )
    affected_sensor_count = min(
        affected_sensor_count,
        len(eligible_sensor_columns),
    )

    affected_sensor_columns = sorted(
        rng.choice(
            eligible_sensor_columns,
            size=affected_sensor_count,
            replace=False,
        ).tolist()
    )

    shift_magnitudes = rng.uniform(
        low=float(magnitude_min),
        high=float(magnitude_max),
        size=affected_sensor_count,
    )

    episode_indices = episode.index.tolist()

    for sensor_column, sigma_multiplier in zip(
        affected_sensor_columns,
        shift_magnitudes,
    ):
        shift_amount = (
            sensor_standard_deviations[sensor_column] * sigma_multiplier
        )
        lots[sensor_column] = lots[sensor_column].astype("float64")
        lots.loc[episode_indices, sensor_column] = (
            lots.loc[episode_indices, sensor_column] + shift_amount
        )

    alarm_evidence_id = "EVID_ALARM_ABRUPT_001"
    affected_sensor_text = ";".join(affected_sensor_columns)
    average_shift_sigma = float(np.mean(shift_magnitudes))

    lots["synthetic_evidence_id"] = lots.get(
        "synthetic_evidence_id",
        "",
    )
    lots["injected_sensor_columns"] = lots.get(
        "injected_sensor_columns",
        "",
    )
    lots["abrupt_shift_sigma"] = lots.get(
        "abrupt_shift_sigma",
        np.nan,
    )

    lots.loc[episode_indices, "is_synthetic_anomaly"] = 1
    lots.loc[
        episode_indices,
        "anomaly_mechanism",
    ] = "abrupt_mean_shift"
    lots.loc[
        episode_indices,
        "root_cause_id",
    ] = "RC_PRESSURE_INSTABILITY"
    lots.loc[
        episode_indices,
        "injected_sensor_columns",
    ] = affected_sensor_text
    lots.loc[
        episode_indices,
        "abrupt_shift_sigma",
    ] = average_shift_sigma
    lot_evidence_ids = {
        str(lot_id): f"EVID_SHIFT_{lot_id}"
        for lot_id in episode["lot_id"].tolist()
    }

    lots.loc[
        episode_indices,
        "synthetic_evidence_id",
    ] = [
        lot_evidence_ids[str(lot_id)]
        for lot_id in episode["lot_id"].tolist()
    ]

    first_lot = episode.iloc[0]
    alarm_start_time = (
        pd.Timestamp(first_lot["event_time"])
        - pd.Timedelta(hours=alarm_lead_hours)
    )

    abrupt_alarm = pd.DataFrame(
        [
            {
                "event_id": "EVT_ABRUPT_001",
                "tool_id": selected_tool,
                "chamber_id": selected_chamber,
                "alarm_code": alarm_code,
                "event_type": "tool_alarm",
                "start_time": alarm_start_time,
                "end_time": alarm_start_time + pd.Timedelta(minutes=30),
                "severity": "high",
                "related_lot_id": first_lot["lot_id"],
                "evidence_id": alarm_evidence_id,
            }
        ]
    )

    updated_tables["tool_events"] = pd.concat(
        [tool_events, abrupt_alarm],
        ignore_index=True,
    )

    rca_records: list[dict[str, Any]] = []

    for case_number, (_, lot) in enumerate(episode.iterrows(), start=1):
        rca_records.append(
            {
                "case_id": f"RCA_ABRUPT_{case_number:03d}",
                "lot_id": lot["lot_id"],
                "root_cause_id": "RC_PRESSURE_INSTABILITY",
                "suspected_cause": (
                    "Synthetic abrupt process mean shift in a correlated "
                    "sensor group."
                ),
                "recommended_action": (
                    "Inspect chamber pressure-control behavior, compare "
                    "adjacent lots, and verify sensor calibration."
                ),
                "outcome": (
                    "Synthetic anomaly injected for controlled evaluation."
                ),
                "evidence_ids": (
                    f"{alarm_evidence_id};"
                    f"{lot_evidence_ids[str(lot['lot_id'])]}"
                ),
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_PRESSURE_INSTABILITY",
            }
        )

    abrupt_rca = pd.DataFrame(rca_records)

    updated_tables["rca_ground_truth"] = pd.concat(
        [rca_ground_truth, abrupt_rca],
        ignore_index=True,
    )
    updated_tables["lots"] = lots

    return updated_tables
def apply_gradual_degradation(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    updated_tables = {
        table_id: table.copy()
        for table_id, table in tables.items()
    }

    mechanism = next(
        item
        for item in config["anomaly_mechanisms"]
        if item["id"] == "gradual_degradation"
    )
    parameters = mechanism["parameters"]

    lots = updated_tables["lots"]
    maintenance = updated_tables["maintenance"]
    rca_ground_truth = updated_tables["rca_ground_truth"]

    injection_rate = float(parameters["injection_rate"])
    sensor_fraction_min, sensor_fraction_max = parameters[
        "affected_sensor_fraction"
    ]
    minimum_window, maximum_window = parameters[
        "degradation_window_lots"
    ]
    magnitude_min, magnitude_max = parameters[
        "maximum_magnitude_sigma"
    ]
    delay_min_days, delay_max_days = parameters[
        "maintenance_delay_days"
    ]
    maintenance_type = str(parameters["maintenance_type"])
    component = str(parameters["component"])

    if not 0 < injection_rate < 1:
        raise ValueError(
            "gradual_degradation injection_rate must be between 0 and 1."
        )

    injection_lot_count = max(
        1,
        int(round(len(lots) * injection_rate)),
    )

    if not minimum_window <= injection_lot_count <= maximum_window:
        raise ValueError(
            "gradual_degradation injection_lot_count must fall within "
            "degradation_window_lots."
        )

    numeric_sensor_columns = [
        column
        for column in lots.columns
        if str(column).startswith("sensor_")
        and pd.api.types.is_numeric_dtype(lots[column])
    ]

    sensor_standard_deviations = lots[numeric_sensor_columns].std(
        axis=0,
        ddof=0,
    )
    eligible_sensor_columns = [
        column
        for column in numeric_sensor_columns
        if np.isfinite(sensor_standard_deviations[column])
        and sensor_standard_deviations[column] > 0
    ]

    if not eligible_sensor_columns:
        raise ValueError(
            "No non-constant numeric sensor columns are available for "
            "gradual_degradation injection."
        )

    used_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in lots.loc[
            lots["anomaly_mechanism"] != "none",
            ["tool_id", "chamber_id"],
        ].itertuples(index=False)
    }

    context_sizes = (
        lots.groupby(["tool_id", "chamber_id"], sort=True)
        .size()
        .reset_index(name="lot_count")
    )
    eligible_contexts = context_sizes.loc[
        context_sizes["lot_count"] >= injection_lot_count
    ].copy()

    eligible_contexts = eligible_contexts.loc[
        ~eligible_contexts.apply(
            lambda row: (
                str(row["tool_id"]),
                str(row["chamber_id"]),
            )
            in used_contexts,
            axis=1,
        )
    ].reset_index(drop=True)

    if eligible_contexts.empty:
        raise ValueError(
            "No unused tool/chamber context contains enough lots for "
            "gradual_degradation."
        )

    rng = np.random.default_rng(config["random_seed"] + 202)

    selected_context = eligible_contexts.iloc[
        rng.integers(low=0, high=len(eligible_contexts))
    ]
    selected_tool = str(selected_context["tool_id"])
    selected_chamber = str(selected_context["chamber_id"])

    episode = (
        lots.loc[
            (lots["tool_id"] == selected_tool)
            & (lots["chamber_id"] == selected_chamber)
        ]
        .sort_values("event_time")
        .head(injection_lot_count)
        .copy()
    )
    episode_indices = episode.index.tolist()

    sensor_fraction = rng.uniform(
        low=float(sensor_fraction_min),
        high=float(sensor_fraction_max),
    )
    affected_sensor_count = max(
        1,
        int(round(len(eligible_sensor_columns) * sensor_fraction)),
    )
    affected_sensor_count = min(
        affected_sensor_count,
        len(eligible_sensor_columns),
    )

    affected_sensor_columns = sorted(
        rng.choice(
            eligible_sensor_columns,
            size=affected_sensor_count,
            replace=False,
        ).tolist()
    )

    final_shift_magnitudes = rng.uniform(
        low=float(magnitude_min),
        high=float(magnitude_max),
        size=affected_sensor_count,
    )
    degradation_progress = np.linspace(
        start=1 / injection_lot_count,
        stop=1.0,
        num=injection_lot_count,
    )

    for sensor_column, final_sigma_multiplier in zip(
        affected_sensor_columns,
        final_shift_magnitudes,
    ):
        final_shift_amount = (
            sensor_standard_deviations[sensor_column]
            * final_sigma_multiplier
        )
        lots[sensor_column] = lots[sensor_column].astype("float64")
        lots.loc[episode_indices, sensor_column] = (
            lots.loc[episode_indices, sensor_column].to_numpy()
            + final_shift_amount * degradation_progress
        )

    maintenance_delay_days = int(
        rng.integers(
            low=int(delay_min_days),
            high=int(delay_max_days) + 1,
        )
    )
    maintenance_evidence_id = "EVID_MAINT_DELAY_GRADUAL_001"
    affected_sensor_text = ";".join(affected_sensor_columns)
    average_final_sigma = float(np.mean(final_shift_magnitudes))

    if "synthetic_evidence_id" not in lots.columns:
        lots["synthetic_evidence_id"] = ""

    if "injected_sensor_columns" not in lots.columns:
        lots["injected_sensor_columns"] = ""

    if "degradation_progress" not in lots.columns:
        lots["degradation_progress"] = np.nan

    if "degradation_final_sigma" not in lots.columns:
        lots["degradation_final_sigma"] = np.nan

    if "maintenance_evidence_id" not in lots.columns:
        lots["maintenance_evidence_id"] = ""

    lot_evidence_ids = {
        str(lot_id): f"EVID_DEGRADATION_{lot_id}"
        for lot_id in episode["lot_id"].tolist()
    }

    lots.loc[episode_indices, "is_synthetic_anomaly"] = 1
    lots.loc[
        episode_indices,
        "anomaly_mechanism",
    ] = "gradual_degradation"
    lots.loc[
        episode_indices,
        "root_cause_id",
    ] = "RC_COMPONENT_WEAR"
    lots.loc[
        episode_indices,
        "injected_sensor_columns",
    ] = affected_sensor_text
    lots.loc[
        episode_indices,
        "degradation_progress",
    ] = degradation_progress
    lots.loc[
        episode_indices,
        "degradation_final_sigma",
    ] = average_final_sigma
    lots.loc[
        episode_indices,
        "maintenance_evidence_id",
    ] = maintenance_evidence_id
    lots.loc[
        episode_indices,
        "synthetic_evidence_id",
    ] = [
        lot_evidence_ids[str(lot_id)]
        for lot_id in episode["lot_id"].tolist()
    ]

    first_lot = episode.iloc[0]
    scheduled_at = (
        pd.Timestamp(first_lot["event_time"])
        - pd.Timedelta(days=maintenance_delay_days)
    )
    performed_at = pd.Timestamp(first_lot["event_time"])

    delayed_maintenance = pd.DataFrame(
        [
            {
                "maintenance_id": "MAINT_GRADUAL_001",
                "tool_id": selected_tool,
                "chamber_id": selected_chamber,
                "scheduled_at": scheduled_at,
                "performed_at": performed_at,
                "delay_days": maintenance_delay_days,
                "maintenance_type": maintenance_type,
                "component": component,
                "replacement_performed": False,
                "related_lot_id": first_lot["lot_id"],
                "evidence_id": maintenance_evidence_id,
            }
        ]
    )

    updated_tables["maintenance"] = pd.concat(
        [maintenance, delayed_maintenance],
        ignore_index=True,
    )

    rca_records: list[dict[str, Any]] = []

    for case_number, (_, lot) in enumerate(episode.iterrows(), start=1):
        rca_records.append(
            {
                "case_id": f"RCA_GRADUAL_{case_number:03d}",
                "lot_id": lot["lot_id"],
                "root_cause_id": "RC_COMPONENT_WEAR",
                "suspected_cause": (
                    "Synthetic gradual component wear after delayed "
                    "preventive inspection."
                ),
                "recommended_action": (
                    "Inspect the pressure-control valve, compare the "
                    "chamber trend before and after the delayed "
                    "maintenance date, and schedule replacement if "
                    "degradation persists."
                ),
                "outcome": (
                    "Synthetic gradual degradation injected for "
                    "controlled evaluation."
                ),
                "evidence_ids": (
                    f"{maintenance_evidence_id};"
                    f"{lot_evidence_ids[str(lot['lot_id'])]}"
                ),
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_COMPONENT_WEAR",
            }
        )

    gradual_rca = pd.DataFrame(rca_records)

    updated_tables["rca_ground_truth"] = pd.concat(
        [rca_ground_truth, gradual_rca],
        ignore_index=True,
    )
    updated_tables["lots"] = lots

    return updated_tables

def apply_variance_instability(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    updated_tables = {
        table_id: table.copy()
        for table_id, table in tables.items()
    }

    mechanism = next(
        item
        for item in config["anomaly_mechanisms"]
        if item["id"] == "variance_instability"
    )
    parameters = mechanism["parameters"]

    lots = updated_tables["lots"]
    tool_events = updated_tables["tool_events"]
    rca_ground_truth = updated_tables["rca_ground_truth"]

    injection_rate = float(parameters["injection_rate"])
    sensor_fraction_min, sensor_fraction_max = parameters[
        "affected_sensor_fraction"
    ]
    multiplier_min, multiplier_max = parameters[
        "variance_multiplier"
    ]
    alarm_code = str(parameters["alarm_code"])
    alarm_lead_hours = float(parameters["alarm_lead_hours"])

    if not 0 < injection_rate < 1:
        raise ValueError(
            "variance_instability injection_rate must be between 0 and 1."
        )

    injection_lot_count = max(
        1,
        int(round(len(lots) * injection_rate)),
    )

    numeric_sensor_columns = [
        column
        for column in lots.columns
        if str(column).startswith("sensor_")
        and pd.api.types.is_numeric_dtype(lots[column])
    ]

    sensor_standard_deviations = lots[numeric_sensor_columns].std(
        axis=0,
        ddof=0,
    )
    eligible_sensor_columns = [
        column
        for column in numeric_sensor_columns
        if np.isfinite(sensor_standard_deviations[column])
        and sensor_standard_deviations[column] > 0
    ]

    if not eligible_sensor_columns:
        raise ValueError(
            "No non-constant numeric sensor columns are available for "
            "variance_instability injection."
        )

    used_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in lots.loc[
            lots["anomaly_mechanism"] != "none",
            ["tool_id", "chamber_id"],
        ].itertuples(index=False)
    }

    context_sizes = (
        lots.groupby(["tool_id", "chamber_id"], sort=True)
        .size()
        .reset_index(name="lot_count")
    )
    eligible_contexts = context_sizes.loc[
        context_sizes["lot_count"] >= injection_lot_count
    ].copy()

    eligible_contexts = eligible_contexts.loc[
        ~eligible_contexts.apply(
            lambda row: (
                str(row["tool_id"]),
                str(row["chamber_id"]),
            )
            in used_contexts,
            axis=1,
        )
    ].reset_index(drop=True)

    if eligible_contexts.empty:
        raise ValueError(
            "No unused tool/chamber context contains enough lots for "
            "variance_instability."
        )

    rng = np.random.default_rng(config["random_seed"] + 303)

    selected_context = eligible_contexts.iloc[
        rng.integers(low=0, high=len(eligible_contexts))
    ]
    selected_tool = str(selected_context["tool_id"])
    selected_chamber = str(selected_context["chamber_id"])

    episode = (
        lots.loc[
            (lots["tool_id"] == selected_tool)
            & (lots["chamber_id"] == selected_chamber)
        ]
        .sort_values("event_time")
        .head(injection_lot_count)
        .copy()
    )
    episode_indices = episode.index.tolist()

    sensor_fraction = rng.uniform(
        low=float(sensor_fraction_min),
        high=float(sensor_fraction_max),
    )
    affected_sensor_count = max(
        1,
        int(round(len(eligible_sensor_columns) * sensor_fraction)),
    )
    affected_sensor_count = min(
        affected_sensor_count,
        len(eligible_sensor_columns),
    )

    affected_sensor_columns = sorted(
        rng.choice(
            eligible_sensor_columns,
            size=affected_sensor_count,
            replace=False,
        ).tolist()
    )

    final_variance_multiplier = float(
        rng.uniform(
            low=float(multiplier_min),
            high=float(multiplier_max),
        )
    )
    variance_progress = np.linspace(
        start=1 / injection_lot_count,
        stop=1.0,
        num=injection_lot_count,
    )
    multiplier_progress = (
        1.0
        + (final_variance_multiplier - 1.0) * variance_progress
    )

    mean_deltas: list[float] = []

    for sensor_column in affected_sensor_columns:
        original_values = (
            lots.loc[episode_indices, sensor_column]
            .astype("float64")
            .to_numpy()
        )
        original_mean = float(np.mean(original_values))
        baseline_std = float(sensor_standard_deviations[sensor_column])

        raw_noise = rng.normal(size=injection_lot_count)
        raw_noise = raw_noise - np.mean(raw_noise)

        raw_noise_std = float(np.std(raw_noise, ddof=0))

        if raw_noise_std == 0:
            raise ValueError(
                "Could not generate non-zero variance noise."
            )

        standardized_noise = raw_noise / raw_noise_std
        shift = (
            standardized_noise
            * baseline_std
            * (final_variance_multiplier - 1.0)
            * variance_progress
        )

        # Re-centre the synthetic disturbance: the episode mean remains
        # unchanged while the variation around that mean increases.
        shift = shift - np.mean(shift)

        lots[sensor_column] = lots[sensor_column].astype("float64")
        lots.loc[episode_indices, sensor_column] = (
            original_values + shift
        )

        updated_mean = float(
            lots.loc[episode_indices, sensor_column].mean()
        )
        mean_deltas.append(updated_mean - original_mean)

    variance_mean_delta = float(
        max(abs(delta) for delta in mean_deltas)
    )

    if "synthetic_evidence_id" not in lots.columns:
        lots["synthetic_evidence_id"] = ""

    if "injected_sensor_columns" not in lots.columns:
        lots["injected_sensor_columns"] = ""

    if "variance_multiplier_progress" not in lots.columns:
        lots["variance_multiplier_progress"] = np.nan

    if "variance_final_multiplier" not in lots.columns:
        lots["variance_final_multiplier"] = np.nan

    if "variance_mean_delta" not in lots.columns:
        lots["variance_mean_delta"] = np.nan

    affected_sensor_text = ";".join(affected_sensor_columns)
    alarm_evidence_id = "EVID_VARIANCE_EVENT_001"

    lot_evidence_ids = {
        str(lot_id): f"EVID_VARIANCE_{lot_id}"
        for lot_id in episode["lot_id"].tolist()
    }

    lots.loc[episode_indices, "is_synthetic_anomaly"] = 1
    lots.loc[
        episode_indices,
        "anomaly_mechanism",
    ] = "variance_instability"
    lots.loc[
        episode_indices,
        "root_cause_id",
    ] = "RC_PROCESS_VARIABILITY"
    lots.loc[
        episode_indices,
        "injected_sensor_columns",
    ] = affected_sensor_text
    lots.loc[
        episode_indices,
        "variance_multiplier_progress",
    ] = multiplier_progress
    lots.loc[
        episode_indices,
        "variance_final_multiplier",
    ] = final_variance_multiplier
    lots.loc[
        episode_indices,
        "variance_mean_delta",
    ] = variance_mean_delta
    lots.loc[
        episode_indices,
        "synthetic_evidence_id",
    ] = [
        lot_evidence_ids[str(lot_id)]
        for lot_id in episode["lot_id"].tolist()
    ]

    first_lot = episode.iloc[0]
    alarm_start_time = (
        pd.Timestamp(first_lot["event_time"])
        - pd.Timedelta(hours=alarm_lead_hours)
    )

    variance_alarm = pd.DataFrame(
        [
            {
                "event_id": "EVT_VARIANCE_001",
                "tool_id": selected_tool,
                "chamber_id": selected_chamber,
                "alarm_code": alarm_code,
                "event_type": "process_variability_warning",
                "start_time": alarm_start_time,
                "end_time": alarm_start_time + pd.Timedelta(minutes=30),
                "severity": "medium",
                "related_lot_id": first_lot["lot_id"],
                "evidence_id": alarm_evidence_id,
            }
        ]
    )

    updated_tables["tool_events"] = pd.concat(
        [tool_events, variance_alarm],
        ignore_index=True,
    )

    rca_records: list[dict[str, Any]] = []

    for case_number, (_, lot) in enumerate(episode.iterrows(), start=1):
        rca_records.append(
            {
                "case_id": f"RCA_VARIANCE_{case_number:03d}",
                "lot_id": lot["lot_id"],
                "root_cause_id": "RC_PROCESS_VARIABILITY",
                "suspected_cause": (
                    "Synthetic variance instability with increasing "
                    "within-context sensor dispersion."
                ),
                "recommended_action": (
                    "Review chamber stability, inspect control-chart "
                    "dispersion, compare adjacent lots, and verify "
                    "process-control settings."
                ),
                "outcome": (
                    "Synthetic variance instability injected for "
                    "controlled evaluation."
                ),
                "evidence_ids": (
                    f"{alarm_evidence_id};"
                    f"{lot_evidence_ids[str(lot['lot_id'])]}"
                ),
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_PROCESS_VARIABILITY",
            }
        )

    variance_rca = pd.DataFrame(rca_records)

    updated_tables["rca_ground_truth"] = pd.concat(
        [rca_ground_truth, variance_rca],
        ignore_index=True,
    )
    updated_tables["lots"] = lots

    return updated_tables


def apply_sensor_faults(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    updated_tables = {
        table_id: table.copy()
        for table_id, table in tables.items()
    }

    mechanism = next(
        item
        for item in config["anomaly_mechanisms"]
        if item["id"] == "sensor_fault"
    )
    parameters = mechanism["parameters"]

    lots = updated_tables["lots"]
    tool_events = updated_tables["tool_events"]
    rca_ground_truth = updated_tables["rca_ground_truth"]

    injection_rate = float(parameters["injection_rate"])
    fault_types = list(parameters["fault_types"])
    sensor_count_min, sensor_count_max = parameters[
        "affected_sensor_count"
    ]
    bias_min_sigma, bias_max_sigma = parameters[
        "calibration_bias_sigma"
    ]
    alarm_lead_hours = float(parameters["alarm_lead_hours"])

    expected_fault_types = {
        "stuck_at",
        "dropout",
        "missing_burst",
        "calibration_bias",
    }

    if set(fault_types) != expected_fault_types:
        raise ValueError(
            "sensor_fault must define exactly: "
            "stuck_at, dropout, missing_burst, calibration_bias."
        )

    if not 0 < injection_rate < 1:
        raise ValueError(
            "sensor_fault injection_rate must be between 0 and 1."
        )

    injection_lot_count = max(
        len(fault_types),
        int(round(len(lots) * injection_rate)),
    )

    numeric_sensor_columns = [
        column
        for column in lots.columns
        if str(column).startswith("sensor_")
        and pd.api.types.is_numeric_dtype(lots[column])
    ]

    sensor_standard_deviations = lots[numeric_sensor_columns].std(
        axis=0,
        ddof=0,
    )
    eligible_sensor_columns = [
        column
        for column in numeric_sensor_columns
        if np.isfinite(sensor_standard_deviations[column])
        and sensor_standard_deviations[column] > 0
    ]

    if len(eligible_sensor_columns) < 2:
        raise ValueError(
            "sensor_fault requires at least two non-constant numeric "
            "sensor columns."
        )

    used_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in lots.loc[
            lots["anomaly_mechanism"] != "none",
            ["tool_id", "chamber_id"],
        ].itertuples(index=False)
    }

    context_sizes = (
        lots.groupby(["tool_id", "chamber_id"], sort=True)
        .size()
        .reset_index(name="lot_count")
    )
    eligible_contexts = context_sizes.loc[
        context_sizes["lot_count"] >= injection_lot_count
    ].copy()

    eligible_contexts = eligible_contexts.loc[
        ~eligible_contexts.apply(
            lambda row: (
                str(row["tool_id"]),
                str(row["chamber_id"]),
            )
            in used_contexts,
            axis=1,
        )
    ].reset_index(drop=True)

    if eligible_contexts.empty:
        raise ValueError(
            "No unused tool/chamber context contains enough lots for "
            "sensor_fault."
        )

    rng = np.random.default_rng(config["random_seed"] + 404)

    selected_context = eligible_contexts.iloc[
        rng.integers(low=0, high=len(eligible_contexts))
    ]
    selected_tool = str(selected_context["tool_id"])
    selected_chamber = str(selected_context["chamber_id"])

    episode = (
        lots.loc[
            (lots["tool_id"] == selected_tool)
            & (lots["chamber_id"] == selected_chamber)
        ]
        .sort_values("event_time")
        .head(injection_lot_count)
        .copy()
    )
    episode_indices = episode.index.tolist()

    if "synthetic_evidence_id" not in lots.columns:
        lots["synthetic_evidence_id"] = ""

    if "injected_sensor_columns" not in lots.columns:
        lots["injected_sensor_columns"] = ""

    if "sensor_fault_type" not in lots.columns:
        lots["sensor_fault_type"] = ""

    if "sensor_fault_detail" not in lots.columns:
        lots["sensor_fault_detail"] = ""

    fault_alarm_codes = {
        "stuck_at": "ALARM_SENSOR_STUCK",
        "dropout": "ALARM_SENSOR_DROPOUT",
        "missing_burst": "ALARM_SENSOR_MISSING_BURST",
        "calibration_bias": "ALARM_SENSOR_CALIBRATION_BIAS",
    }

    fault_actions = {
        "stuck_at": (
            "Inspect the sensor signal path and replace or reset the "
            "stuck measurement channel."
        ),
        "dropout": (
            "Inspect intermittent data acquisition, communication, and "
            "sensor connector integrity."
        ),
        "missing_burst": (
            "Investigate the sustained missing-data interval and verify "
            "data logger and sensor availability."
        ),
        "calibration_bias": (
            "Recalibrate the sensor and compare its offset against a "
            "reference measurement."
        ),
    }

    fault_groups = np.array_split(
        np.arange(injection_lot_count),
        len(fault_types),
    )

    fault_events: list[dict[str, Any]] = []
    rca_records: list[dict[str, Any]] = []

    for fault_number, (fault_type, positions) in enumerate(
        zip(fault_types, fault_groups),
        start=1,
    ):
        group_indices = [
            episode_indices[position]
            for position in positions.tolist()
        ]
        group_lots = lots.loc[group_indices].copy()

        requested_sensor_count = int(
            rng.integers(
                low=int(sensor_count_min),
                high=int(sensor_count_max) + 1,
            )
        )

        if fault_type == "dropout":
            requested_sensor_count = max(
                2,
                requested_sensor_count,
            )

        affected_sensor_count = min(
            requested_sensor_count,
            len(eligible_sensor_columns),
        )

        affected_sensor_columns = sorted(
            rng.choice(
                eligible_sensor_columns,
                size=affected_sensor_count,
                replace=False,
            ).tolist()
        )
        affected_sensor_text = ";".join(affected_sensor_columns)

        fault_event_id = (
            f"EVT_SENSOR_{fault_type.upper()}_{fault_number:03d}"
        )
        fault_event_evidence_id = (
            f"EVID_SENSOR_EVENT_{fault_type.upper()}_{fault_number:03d}"
        )
        first_lot = group_lots.iloc[0]
        alarm_start_time = (
            pd.Timestamp(first_lot["event_time"])
            - pd.Timedelta(hours=alarm_lead_hours)
        )

        if fault_type == "stuck_at":
            for sensor_column in affected_sensor_columns:
                stuck_value = float(
                    lots.loc[group_indices[0], sensor_column]
                )
                lots[sensor_column] = lots[sensor_column].astype(
                    "float64"
                )
                lots.loc[group_indices, sensor_column] = stuck_value

            fault_detail = "constant_stuck_at_value"

        elif fault_type == "dropout":
            for row_number, lot_index in enumerate(group_indices):
                sensor_column = affected_sensor_columns[
                    row_number % len(affected_sensor_columns)
                ]
                lots.loc[lot_index, sensor_column] = np.nan

            fault_detail = "intermittent_single_sensor_dropout"

        elif fault_type == "missing_burst":
            for sensor_column in affected_sensor_columns:
                lots.loc[group_indices, sensor_column] = np.nan

            fault_detail = "consecutive_missing_data_burst"

        elif fault_type == "calibration_bias":
            bias_sigma = float(
                rng.uniform(
                    low=float(bias_min_sigma),
                    high=float(bias_max_sigma),
                )
            )

            for sensor_column in affected_sensor_columns:
                bias_amount = (
                    sensor_standard_deviations[sensor_column] * bias_sigma
                )
                lots[sensor_column] = lots[sensor_column].astype(
                    "float64"
                )
                lots.loc[group_indices, sensor_column] = (
                    lots.loc[group_indices, sensor_column]
                    + bias_amount
                )

            fault_detail = f"positive_calibration_bias_sigma={bias_sigma:.3f}"

        else:
            raise ValueError(
                f"Unsupported sensor fault type: {fault_type}"
            )

        lot_evidence_ids = {
            str(lot_id): (
                f"EVID_SENSOR_{fault_type.upper()}_{lot_id}"
            )
            for lot_id in group_lots["lot_id"].tolist()
        }

        lots.loc[group_indices, "is_synthetic_anomaly"] = 1
        lots.loc[group_indices, "anomaly_mechanism"] = "sensor_fault"
        lots.loc[
            group_indices,
            "root_cause_id",
        ] = "RC_SENSOR_HARDWARE"
        lots.loc[
            group_indices,
            "sensor_fault_type",
        ] = fault_type
        lots.loc[
            group_indices,
            "sensor_fault_detail",
        ] = fault_detail
        lots.loc[
            group_indices,
            "injected_sensor_columns",
        ] = affected_sensor_text
        lots.loc[
            group_indices,
            "synthetic_evidence_id",
        ] = [
            lot_evidence_ids[str(lot_id)]
            for lot_id in group_lots["lot_id"].tolist()
        ]

        fault_events.append(
            {
                "event_id": fault_event_id,
                "tool_id": selected_tool,
                "chamber_id": selected_chamber,
                "alarm_code": fault_alarm_codes[fault_type],
                "event_type": "sensor_fault_alarm",
                "start_time": alarm_start_time,
                "end_time": alarm_start_time + pd.Timedelta(minutes=30),
                "severity": "high",
                "related_lot_id": first_lot["lot_id"],
                "evidence_id": fault_event_evidence_id,
            }
        )

        for _, lot in group_lots.iterrows():
            rca_records.append(
                {
                    "case_id": (
                        f"RCA_SENSOR_{fault_type.upper()}_"
                        f"{lot['lot_id']}"
                    ),
                    "lot_id": lot["lot_id"],
                    "root_cause_id": "RC_SENSOR_HARDWARE",
                    "suspected_cause": (
                        f"Synthetic {fault_type} sensor hardware fault."
                    ),
                    "recommended_action": fault_actions[fault_type],
                    "outcome": (
                        "Synthetic sensor fault injected for controlled "
                        "evaluation."
                    ),
                    "evidence_ids": (
                        f"{fault_event_evidence_id};"
                        f"{lot_evidence_ids[str(lot['lot_id'])]}"
                    ),
                    "supports_abstention": False,
                    "top3_acceptable_causes": "RC_SENSOR_HARDWARE",
                }
            )

    sensor_fault_events = pd.DataFrame(fault_events)
    sensor_fault_rca = pd.DataFrame(rca_records)

    updated_tables["tool_events"] = pd.concat(
        [tool_events, sensor_fault_events],
        ignore_index=True,
    )
    updated_tables["rca_ground_truth"] = pd.concat(
        [rca_ground_truth, sensor_fault_rca],
        ignore_index=True,
    )
    updated_tables["lots"] = lots

    return updated_tables



def validate_generated_tables(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> None:
    output_specs = {
        output["id"]: output
        for output in config["outputs"]
    }

    expected_table_ids = set(output_specs)

    if set(tables) != expected_table_ids:
        raise ValueError(
            "Generated tables do not match the configured outputs. "
            f"Expected {sorted(expected_table_ids)}, got {sorted(tables)}."
        )

    for table_id, table in tables.items():
        required_columns = set(output_specs[table_id]["required_columns"])
        missing_columns = required_columns.difference(table.columns)

        if missing_columns:
            raise ValueError(
                f"Generated table '{table_id}' is missing columns: "
                f"{sorted(missing_columns)}"
            )

    lots = tables["lots"]

    if not lots["lot_id"].is_unique:
        raise ValueError("Generated lot_id values must be unique.")

    if not set(lots["quality_label"].unique()).issubset({0, 1}):
        raise ValueError("quality_label must contain only 0 or 1.")

    if not (lots["label_available_at"] > lots["event_time"]).all():
        raise ValueError(
            "Every label_available_at must occur after event_time."
        )

    allowed_mechanisms = {
        "none",
        "abrupt_mean_shift",
        "gradual_degradation",
        "variance_instability",
        "sensor_fault",
    }

    if not set(lots["anomaly_mechanism"].unique()).issubset(
        allowed_mechanisms
    ):
        raise ValueError(
            "Only 'none' and 'abrupt_mean_shift' are allowed in this "
            "checkpoint."
        )

    abrupt_lots = lots.loc[
        lots["anomaly_mechanism"] == "abrupt_mean_shift"
    ]

    if abrupt_lots.empty:
        return

    if not abrupt_lots["is_synthetic_anomaly"].eq(1).all():
        raise ValueError(
            "abrupt_mean_shift lots must have is_synthetic_anomaly=1."
        )

    if not abrupt_lots["root_cause_id"].eq(
        "RC_PRESSURE_INSTABILITY"
    ).all():
        raise ValueError(
            "abrupt_mean_shift lots must use "
            "RC_PRESSURE_INSTABILITY."
        )

    required_abrupt_columns = {
        "synthetic_evidence_id",
        "injected_sensor_columns",
        "abrupt_shift_sigma",
    }

    missing_abrupt_columns = required_abrupt_columns.difference(
        abrupt_lots.columns
    )

    if missing_abrupt_columns:
        raise ValueError(
            "abrupt_mean_shift lots are missing required metadata: "
            f"{sorted(missing_abrupt_columns)}"
        )

    if abrupt_lots["synthetic_evidence_id"].eq("").any():
        raise ValueError(
            "abrupt_mean_shift lots must contain evidence IDs."
        )

    abrupt_events = tables["tool_events"].loc[
        tables["tool_events"]["event_id"] == "EVT_ABRUPT_001"
    ]

    if len(abrupt_events) != 1:
        raise ValueError(
            "Exactly one abrupt_mean_shift tool-alarm event is required."
        )

    abrupt_cases = tables["rca_ground_truth"].loc[
        tables["rca_ground_truth"]["root_cause_id"]
        == "RC_PRESSURE_INSTABILITY"
    ]

    if len(abrupt_cases) != len(abrupt_lots):
        raise ValueError(
            "Every abrupt_mean_shift lot must have one RCA ground-truth "
            "case."
        )
    gradual_lots = lots.loc[
        lots["anomaly_mechanism"] == "gradual_degradation"
    ]

    if gradual_lots.empty:
        return

    if not gradual_lots["is_synthetic_anomaly"].eq(1).all():
        raise ValueError(
            "gradual_degradation lots must have "
            "is_synthetic_anomaly=1."
        )

    if not gradual_lots["root_cause_id"].eq(
        "RC_COMPONENT_WEAR"
    ).all():
        raise ValueError(
            "gradual_degradation lots must use RC_COMPONENT_WEAR."
        )

    required_gradual_columns = {
        "synthetic_evidence_id",
        "injected_sensor_columns",
        "degradation_progress",
        "degradation_final_sigma",
        "maintenance_evidence_id",
    }

    missing_gradual_columns = required_gradual_columns.difference(
        gradual_lots.columns
    )

    if missing_gradual_columns:
        raise ValueError(
            "gradual_degradation lots are missing metadata: "
            f"{sorted(missing_gradual_columns)}"
        )

    if gradual_lots["synthetic_evidence_id"].eq("").any():
        raise ValueError(
            "gradual_degradation lots must contain evidence IDs."
        )

    if gradual_lots["maintenance_evidence_id"].ne(
        "EVID_MAINT_DELAY_GRADUAL_001"
    ).any():
        raise ValueError(
            "gradual_degradation lots must reference the delayed "
            "maintenance evidence ID."
        )

    ordered_progress = gradual_lots.sort_values(
        "event_time"
    )["degradation_progress"]

    if not ordered_progress.is_monotonic_increasing:
        raise ValueError(
            "gradual_degradation progress must increase over time."
        )

    abrupt_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in abrupt_lots[
            ["tool_id", "chamber_id"]
        ].drop_duplicates().itertuples(index=False)
    }
    gradual_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in gradual_lots[
            ["tool_id", "chamber_id"]
        ].drop_duplicates().itertuples(index=False)
    }

    if abrupt_contexts.intersection(gradual_contexts):
        raise ValueError(
            "gradual_degradation must use a different tool/chamber "
            "context from abrupt_mean_shift."
        )

    delayed_maintenance = tables["maintenance"].loc[
        tables["maintenance"]["maintenance_id"]
        == "MAINT_GRADUAL_001"
    ]

    if len(delayed_maintenance) != 1:
        raise ValueError(
            "Exactly one delayed-maintenance record is required."
        )

    if int(delayed_maintenance.iloc[0]["delay_days"]) <= 0:
        raise ValueError(
            "gradual_degradation maintenance delay must be positive."
        )

    gradual_cases = tables["rca_ground_truth"].loc[
        tables["rca_ground_truth"]["root_cause_id"]
        == "RC_COMPONENT_WEAR"
    ]

    if len(gradual_cases) != len(gradual_lots):
        raise ValueError(
            "Every gradual_degradation lot must have one RCA "
            "ground-truth case."
        )

def write_generated_tables(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    repo_root: Path,
    output_dir: Path | None,
) -> list[Path]:
    output_specs = {
        output["id"]: output
        for output in config["outputs"]
    }

    written_paths: list[Path] = []

    for table_id, table in tables.items():
        configured_path = Path(output_specs[table_id]["path"])

        if output_dir is None:
            target_path = repo_root / configured_path
        else:
            target_path = output_dir / configured_path.name

        target_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target_path, index=False)
        written_paths.append(target_path)

    return written_paths

def validate_variance_instability(
    tables: dict[str, pd.DataFrame],
) -> None:
    lots = tables["lots"]

    variance_lots = lots.loc[
        lots["anomaly_mechanism"] == "variance_instability"
    ].copy()

    if variance_lots.empty:
        raise ValueError(
            "variance_instability must inject at least one lot."
        )

    if not variance_lots["is_synthetic_anomaly"].eq(1).all():
        raise ValueError(
            "variance_instability lots must have "
            "is_synthetic_anomaly=1."
        )

    if not variance_lots["root_cause_id"].eq(
        "RC_PROCESS_VARIABILITY"
    ).all():
        raise ValueError(
            "variance_instability lots must use "
            "RC_PROCESS_VARIABILITY."
        )

    required_columns = {
        "synthetic_evidence_id",
        "injected_sensor_columns",
        "variance_multiplier_progress",
        "variance_final_multiplier",
        "variance_mean_delta",
    }

    missing_columns = required_columns.difference(
        variance_lots.columns
    )

    if missing_columns:
        raise ValueError(
            "variance_instability lots are missing metadata: "
            f"{sorted(missing_columns)}"
        )

    if variance_lots["synthetic_evidence_id"].fillna("").eq("").any():
        raise ValueError(
            "variance_instability lots must contain evidence IDs."
        )

    ordered_multiplier = variance_lots.sort_values(
        "event_time"
    )["variance_multiplier_progress"]

    if not ordered_multiplier.is_monotonic_increasing:
        raise ValueError(
            "variance multiplier must increase over time."
        )

    if variance_lots["variance_final_multiplier"].le(1.0).any():
        raise ValueError(
            "variance final multiplier must be greater than 1.0."
        )

    if variance_lots["variance_mean_delta"].abs().gt(1e-9).any():
        raise ValueError(
            "variance_instability must preserve the episode mean."
        )

    earlier_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in lots.loc[
            lots["anomaly_mechanism"].isin(
                {"abrupt_mean_shift", "gradual_degradation"}
            ),
            ["tool_id", "chamber_id"],
        ].drop_duplicates().itertuples(index=False)
    }
    variance_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in variance_lots[
            ["tool_id", "chamber_id"]
        ].drop_duplicates().itertuples(index=False)
    }

    if earlier_contexts.intersection(variance_contexts):
        raise ValueError(
            "variance_instability must use a context not used by "
            "earlier anomaly mechanisms."
        )

    variance_events = tables["tool_events"].loc[
        tables["tool_events"]["event_id"] == "EVT_VARIANCE_001"
    ]

    if len(variance_events) != 1:
        raise ValueError(
            "Exactly one variance-instability tool event is required."
        )

    variance_cases = tables["rca_ground_truth"].loc[
        tables["rca_ground_truth"]["root_cause_id"]
        == "RC_PROCESS_VARIABILITY"
    ]

    if len(variance_cases) != len(variance_lots):
        raise ValueError(
            "Every variance_instability lot must have one RCA "
            "ground-truth case."
        )

def validate_sensor_faults(
    tables: dict[str, pd.DataFrame],
) -> None:
    lots = tables["lots"]

    sensor_fault_lots = lots.loc[
        lots["anomaly_mechanism"] == "sensor_fault"
    ].copy()

    if sensor_fault_lots.empty:
        raise ValueError(
            "sensor_fault must inject at least one lot."
        )

    expected_fault_types = {
        "stuck_at",
        "dropout",
        "missing_burst",
        "calibration_bias",
    }

    actual_fault_types = set(
        sensor_fault_lots["sensor_fault_type"].unique()
    )

    if actual_fault_types != expected_fault_types:
        raise ValueError(
            "sensor_fault must include all four fault types."
        )

    if not sensor_fault_lots["is_synthetic_anomaly"].eq(1).all():
        raise ValueError(
            "sensor_fault lots must have is_synthetic_anomaly=1."
        )

    if not sensor_fault_lots["root_cause_id"].eq(
        "RC_SENSOR_HARDWARE"
    ).all():
        raise ValueError(
            "sensor_fault lots must use RC_SENSOR_HARDWARE."
        )

    required_columns = {
        "synthetic_evidence_id",
        "injected_sensor_columns",
        "sensor_fault_type",
        "sensor_fault_detail",
    }

    missing_columns = required_columns.difference(
        sensor_fault_lots.columns
    )

    if missing_columns:
        raise ValueError(
            "sensor_fault lots are missing metadata: "
            f"{sorted(missing_columns)}"
        )

    if sensor_fault_lots["synthetic_evidence_id"].fillna("").eq("").any():
        raise ValueError(
            "sensor_fault lots must contain evidence IDs."
        )

    earlier_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in lots.loc[
            lots["anomaly_mechanism"].isin(
                {
                    "abrupt_mean_shift",
                    "gradual_degradation",
                    "variance_instability",
                }
            ),
            ["tool_id", "chamber_id"],
        ].drop_duplicates().itertuples(index=False)
    }
    sensor_fault_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in sensor_fault_lots[
            ["tool_id", "chamber_id"]
        ].drop_duplicates().itertuples(index=False)
    }

    if earlier_contexts.intersection(sensor_fault_contexts):
        raise ValueError(
            "sensor_fault must use a context not used by earlier "
            "anomaly mechanisms."
        )

    sensor_fault_events = tables["tool_events"].loc[
        tables["tool_events"]["event_type"] == "sensor_fault_alarm"
    ]

    if len(sensor_fault_events) != 4:
        raise ValueError(
            "sensor_fault must create exactly four tool events."
        )

    sensor_fault_cases = tables["rca_ground_truth"].loc[
        tables["rca_ground_truth"]["root_cause_id"]
        == "RC_SENSOR_HARDWARE"
    ]

    if len(sensor_fault_cases) != len(sensor_fault_lots):
        raise ValueError(
            "Every sensor_fault lot must have one RCA ground-truth case."
        )

    for fault_type in expected_fault_types:
        type_lots = sensor_fault_lots.loc[
            sensor_fault_lots["sensor_fault_type"] == fault_type
        ]

        if type_lots.empty:
            raise ValueError(
                f"sensor_fault type '{fault_type}' has no lots."
            )

        type_events = sensor_fault_events.loc[
            sensor_fault_events["event_id"].str.contains(
                fault_type.upper(),
                regex=False,
            )
        ]

        if len(type_events) != 1:
            raise ValueError(
                f"sensor_fault type '{fault_type}' must have one event."
            )

    missing_burst_lots = sensor_fault_lots.loc[
        sensor_fault_lots["sensor_fault_type"] == "missing_burst"
    ]

    missing_burst_columns = set(
        ";".join(
            missing_burst_lots["injected_sensor_columns"].tolist()
        ).split(";")
    )

    has_true_missing_burst = any(
        missing_burst_lots[column].isna().all()
        for column in missing_burst_columns
        if column
    )

    if not has_true_missing_burst:
        raise ValueError(
            "missing_burst must create a consecutive missing-value burst."
        )

    stuck_at_lots = sensor_fault_lots.loc[
        sensor_fault_lots["sensor_fault_type"] == "stuck_at"
    ]

    stuck_at_columns = set(
        ";".join(
            stuck_at_lots["injected_sensor_columns"].tolist()
        ).split(";")
    )

    has_stuck_sensor = any(
        stuck_at_lots[column].dropna().nunique() == 1
        for column in stuck_at_columns
        if column
    )

    if not has_stuck_sensor:
        raise ValueError(
            "stuck_at must produce at least one constant sensor value."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic baseline process context for "
            "WaferWatch Synthetic Data V2."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_data_v2.json"),
        help="Path to the Synthetic Data V2 configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory override. By default, paths in the "
            "configuration are used."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path.cwd().resolve()
    config_path = args.config.resolve()

    try:
        config = load_config(config_path)
        validate_config(
            config=config,
            repo_root=repo_root,
            check_input_path=True,
        )

        input_path = repo_root / config["provenance"]["input_path"]
        base_frame = pd.read_csv(input_path)

        tables = build_baseline_tables(
            base_frame=base_frame,
            config=config,
        )
        tables = apply_abrupt_mean_shift(
            tables=tables,
            config=config,
        )
        tables = apply_gradual_degradation(
            tables=tables,
            config=config,
        )
        tables = apply_variance_instability(
            tables=tables,
            config=config,
        )
        tables = apply_sensor_faults(
            tables=tables,
            config=config,
        )        
        validate_generated_tables(
            tables=tables,
            config=config,
        )
        validate_variance_instability(
            tables=tables,
        )
        validate_sensor_faults(
            tables=tables,
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else None
        )
        written_paths = write_generated_tables(
            tables=tables,
            config=config,
            repo_root=repo_root,
            output_dir=output_dir,
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        SyntheticV2ConfigError,
    ) as error:
        print("SYNTHETIC_V2_CONTEXT_GENERATION_FAILED")
        print(f"- {error}")
        return 1

    lots = tables["lots"]
    unseen_lot_count = int(lots["is_unseen_context"].sum())

    print("SYNTHETIC_V2_CONTEXT_GENERATION_OK")
    print(f"Input rows: {len(base_frame)}")
    print(f"Sensor columns retained: {base_frame.shape[1] - 1}")
    print(f"Generated lots: {len(lots)}")
    print(f"Unseen-context lots: {unseen_lot_count}")
    abrupt_lot_count = int(
        lots["anomaly_mechanism"].eq("abrupt_mean_shift").sum()
    )
    gradual_lot_count = int(
        lots["anomaly_mechanism"].eq("gradual_degradation").sum()
    )
    variance_lot_count = int(
        lots["anomaly_mechanism"].eq("variance_instability").sum()
    )
    sensor_fault_lot_count = int(
        lots["anomaly_mechanism"].eq("sensor_fault").sum()
    )
    print(
        f"Injected synthetic anomalies: "
        f"{int(lots['is_synthetic_anomaly'].sum())}"
    )
    print(f"Abrupt mean-shift lots: {abrupt_lot_count}")
    print(f"Gradual-degradation lots: {gradual_lot_count}")
    print(f"Variance-instability lots: {variance_lot_count}")
    print(f"Sensor-fault lots: {sensor_fault_lot_count}")
    print(f"RCA ground-truth cases: {len(tables['rca_ground_truth'])}")
    print("Written files:")

    for written_path in written_paths:
        print(f"- {written_path}")

    print(
        "Implemented mechanisms: abrupt_mean_shift, "
        "gradual_degradation, variance_instability, sensor_fault"
    )
    print(
        "All other Synthetic Data V2 anomaly mechanisms remain disabled."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())