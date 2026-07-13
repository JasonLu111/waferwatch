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

    allowed_mechanisms = {"none", "abrupt_mean_shift"}

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
        
        validate_generated_tables(
            tables=tables,
            config=config,
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

    print(
        f"Injected synthetic anomalies: "
        f"{int(lots['is_synthetic_anomaly'].sum())}"
    )
    print(f"Abrupt mean-shift lots: {abrupt_lot_count}")
    print(f"RCA ground-truth cases: {len(tables['rca_ground_truth'])}")
    print("Written files:")

    for written_path in written_paths:
        print(f"- {written_path}")

    print("Implemented mechanism: abrupt_mean_shift")
    print(
        "All other Synthetic Data V2 anomaly mechanisms remain disabled."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())