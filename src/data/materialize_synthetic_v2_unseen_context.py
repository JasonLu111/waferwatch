"""Materialize the UNSEEN_TOOL_CHAMBER counterfactual stress-test overlay.

This module preserves core Synthetic Data V2 outputs. It creates a separate
research-only held-out TOOL_08 evaluation artifact by transferring existing
synthetic anomaly templates to unseen-context host lots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_OUTPUT_FILES = {
    "lots": "data/synthetic/v2/synthetic_secom_v2.csv",
    "tool_events": "data/synthetic/v2/synthetic_tool_events.csv",
    "maintenance": "data/synthetic/v2/synthetic_maintenance.csv",
    "process_changes": (
        "data/synthetic/v2/synthetic_process_changes.csv"
    ),
    "rca_ground_truth": (
        "data/synthetic/v2/synthetic_rca_ground_truth.csv"
    ),
}

OUTPUT_FILES = {
    "train_lots": "unseen_tool_chamber_train_lots.csv",
    "eval_lots": "unseen_tool_chamber_eval_lots.csv",
    "tool_events": "unseen_tool_chamber_eval_tool_events.csv",
    "maintenance": "unseen_tool_chamber_eval_maintenance.csv",
    "process_changes": (
        "unseen_tool_chamber_eval_process_changes.csv"
    ),
    "rca_ground_truth": (
        "unseen_tool_chamber_eval_rca_ground_truth.csv"
    ),
    "provenance": "unseen_tool_chamber_overlay_provenance.csv",
    "manifest": "unseen_tool_chamber_manifest.json",
}

EVENT_BACKED_MECHANISMS = {
    "abrupt_mean_shift",
    "sensor_fault",
    "contextual_anomaly",
}


class UnseenContextMaterializationError(ValueError):
    """Raised when the held-out context overlay contract is invalid."""


@dataclass(frozen=True)
class UnseenContextSummary:
    training_lot_count: int
    evaluation_lot_count: int
    synthetic_anomaly_count: int
    achieved_anomaly_rate: float
    mechanism_counts: dict[str, int]
    output_dir: Path


def _fail(message: str) -> None:
    raise UnseenContextMaterializationError(message)


def _true_mask(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
    )


def _non_empty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _evidence_tokens(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()

    return {
        token.strip()
        for token in str(value).split(";")
        if token.strip()
    }


def _parse_sensor_columns(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    return sorted(
        {
            item.strip()
            for item in str(value)
            .replace("|", ";")
            .replace(",", ";")
            .split(";")
            if item.strip()
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_tables(repo_root: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    for table_id, relative_path in BASE_OUTPUT_FILES.items():
        path = repo_root / relative_path

        if not path.exists():
            _fail(f"Missing Synthetic Data V2 output: {path}")

        tables[table_id] = pd.read_csv(path)

    return tables


def _load_scenario(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(
        repo_root
        / "configs"
        / "synthetic_data_v2_stress_test_manifest.json"
    )
    scenario = next(
        (
            item
            for item in manifest["scenarios"]
            if item["id"] == "UNSEEN_TOOL_CHAMBER"
        ),
        None,
    )

    if scenario is None:
        _fail("UNSEEN_TOOL_CHAMBER is missing from the stress manifest.")

    return manifest, scenario


def _contract(scenario: dict[str, Any]) -> dict[str, Any]:
    materialization = scenario.get("materialization", {})
    required_keys = {
        "mode",
        "base_evaluation_selector",
        "target_anomaly_count",
        "achieved_anomaly_rate",
        "include_contextual_control",
        "preserve_core_v2_outputs",
        "mechanism_plan",
        "disclaimer",
    }
    missing_keys = required_keys.difference(materialization)

    if missing_keys:
        _fail(
            "UNSEEN_TOOL_CHAMBER materialization is missing: "
            f"{sorted(missing_keys)}"
        )

    if materialization["mode"] != (
        "counterfactual_unseen_context_overlay"
    ):
        _fail("Unexpected unseen-context materialization mode.")

    if materialization["preserve_core_v2_outputs"] is not True:
        _fail(
            "UNSEEN_TOOL_CHAMBER must preserve core V2 outputs."
        )

    return materialization


def _scenario_seed(manifest: dict[str, Any]) -> int:
    base_seed = int(manifest["research_scope"]["random_seed"])
    return base_seed + sum(
        (index + 1) * ord(character)
        for index, character in enumerate(
            "UNSEEN_TOOL_CHAMBER"
        )
    )


def _empty_row(
    frame: pd.DataFrame,
    values: dict[str, Any],
) -> dict[str, Any]:
    row = {
        column: ""
        for column in frame.columns
    }

    for column, value in values.items():
        if column in row:
            row[column] = value

    return row


def _template_sensor_columns(
    template: pd.Series,
    mechanism: str,
) -> list[str]:
    if mechanism == "contextual_anomaly":
        sensor_columns = _parse_sensor_columns(
            template.get("contextual_sensor_columns")
        )
    else:
        sensor_columns = _parse_sensor_columns(
            template.get("injected_sensor_columns")
        )

    if not sensor_columns:
        _fail(
            f"Template {template['lot_id']} has no injected sensors."
        )

    return sensor_columns


def _select_template(
    seen_anomalies: pd.DataFrame,
    mechanism: str,
    sensor_fault_occurrence: int,
) -> pd.Series:
    pool = seen_anomalies.loc[
        seen_anomalies["anomaly_mechanism"].eq(mechanism)
    ].copy()

    if mechanism == "sensor_fault":
        fault_types = sorted(
            pool["sensor_fault_type"]
            .fillna("")
            .astype(str)
            .unique()
        )
        fault_types = [
            fault_type
            for fault_type in fault_types
            if fault_type
        ]

        if len(fault_types) < sensor_fault_occurrence:
            _fail(
                "Not enough sensor-fault subtypes for unseen overlay."
            )

        target_fault_type = fault_types[
            sensor_fault_occurrence - 1
        ]
        pool = pool.loc[
            pool["sensor_fault_type"]
            .astype(str)
            .eq(target_fault_type)
        ]

    if pool.empty:
        _fail(
            f"No seen anomaly template available for {mechanism}."
        )

    return pool.sort_values(
        ["event_time", "lot_id"]
    ).iloc[0]


def _copy_template_to_host(
    evaluation_lots: pd.DataFrame,
    host_index: int,
    template: pd.Series,
    mechanism: str,
    overlay_id: str,
) -> tuple[pd.DataFrame, list[str]]:
    sensor_columns = _template_sensor_columns(
        template=template,
        mechanism=mechanism,
    )

    missing_sensor_columns = set(sensor_columns).difference(
        evaluation_lots.columns
    )

    if missing_sensor_columns:
        _fail(
            "Missing sensor columns in unseen host data: "
            f"{sorted(missing_sensor_columns)}"
        )

    for column in sensor_columns:
        evaluation_lots.loc[host_index, column] = template[column]

    metadata_columns = {
        "anomaly_mechanism",
        "root_cause_id",
        "injected_sensor_columns",
        "abrupt_shift_sigma",
        "degradation_progress",
        "degradation_final_sigma",
        "variance_multiplier_progress",
        "variance_final_multiplier",
        "variance_mean_delta",
        "sensor_fault_type",
        "contextual_normal_recipe",
        "contextual_anomalous_recipe",
        "contextual_sensor_columns",
    }

    for column in metadata_columns:
        if column in evaluation_lots.columns and column in template.index:
            evaluation_lots.loc[host_index, column] = template[column]

    evaluation_lots.loc[
        host_index,
        "is_synthetic_anomaly",
    ] = 1
    evaluation_lots.loc[
        host_index,
        "anomaly_mechanism",
    ] = mechanism
    evaluation_lots.loc[
        host_index,
        "root_cause_id",
    ] = template["root_cause_id"]
    evaluation_lots.loc[
        host_index,
        "is_benign_drift",
    ] = 0
    evaluation_lots.loc[
        host_index,
        "synthetic_evidence_id",
    ] = f"EVID_UNSEEN_OVERLAY_{overlay_id}"
    evaluation_lots.loc[
        host_index,
        "counterfactual_overlay_id",
    ] = overlay_id
    evaluation_lots.loc[
        host_index,
        "counterfactual_overlay_mode",
    ] = "counterfactual_unseen_context_overlay"
    evaluation_lots.loc[
        host_index,
        "counterfactual_template_lot_id",
    ] = template["lot_id"]

    if mechanism != "gradual_degradation":
        evaluation_lots.loc[
            host_index,
            "maintenance_evidence_id",
        ] = ""

    if mechanism != "contextual_anomaly":
        evaluation_lots.loc[
            host_index,
            "is_contextual_control",
        ] = 0
        evaluation_lots.loc[
            host_index,
            "contextual_pair_id",
        ] = ""
        evaluation_lots.loc[
            host_index,
            "contextual_expected_status",
        ] = ""
        evaluation_lots.loc[
            host_index,
            "recipe_context_evidence_id",
        ] = ""

    return evaluation_lots, sensor_columns


def _build_overlay(
    source_tables: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, int],
    int,
]:
    contract = _contract(scenario)
    lots = source_tables["lots"].copy()
    seed = _scenario_seed(manifest)

    unseen_normal_lots = lots.loc[
        _true_mask(lots["is_unseen_context"])
        & ~_true_mask(lots["is_synthetic_anomaly"])
    ].copy()

    seen_lots = lots.loc[
        ~_true_mask(lots["is_unseen_context"])
    ].copy()
    seen_anomalies = seen_lots.loc[
        _true_mask(seen_lots["is_synthetic_anomaly"])
    ].copy()

    if unseen_normal_lots.empty:
        _fail("No normal unseen-context host lots are available.")

    mechanism_plan = [
        str(mechanism)
        for mechanism in contract["mechanism_plan"]
    ]
    target_anomaly_count = int(
        contract["target_anomaly_count"]
    )

    if len(mechanism_plan) != target_anomaly_count:
        _fail(
            "mechanism_plan length must equal target_anomaly_count."
        )

    host_count = target_anomaly_count + 1
    if len(unseen_normal_lots) < host_count:
        _fail(
            "Not enough unseen normal lots for anomalies and one "
            "contextual control."
        )

    selected_hosts = unseen_normal_lots.sample(
        n=host_count,
        random_state=seed,
    ).sort_values(["event_time", "lot_id"])

    anomaly_host_indices = selected_hosts.iloc[
        :target_anomaly_count
    ].index.to_list()
    contextual_control_index = selected_hosts.iloc[
        target_anomaly_count
    ].name

    evaluation_lots = unseen_normal_lots.copy()

    new_columns = {
        "counterfactual_overlay_id": "",
        "counterfactual_overlay_mode": "",
        "counterfactual_template_lot_id": "",
    }

    for column, default_value in new_columns.items():
        if column not in evaluation_lots.columns:
            evaluation_lots[column] = default_value

    overlay_rows: list[dict[str, Any]] = []
    rca_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    maintenance_rows: list[dict[str, Any]] = []

    tool_events_template = source_tables["tool_events"]
    maintenance_template = source_tables["maintenance"]
    rca_template = source_tables["rca_ground_truth"]

    sensor_fault_occurrence = 0
    contextual_template: pd.Series | None = None
    contextual_sensor_columns: list[str] = []
    contextual_overlay_id = ""
    mechanism_counts: dict[str, int] = {}

    for sequence, (
        mechanism,
        host_index,
    ) in enumerate(
        zip(mechanism_plan, anomaly_host_indices),
        start=1,
    ):
        if mechanism == "sensor_fault":
            sensor_fault_occurrence += 1
            template = _select_template(
                seen_anomalies=seen_anomalies,
                mechanism=mechanism,
                sensor_fault_occurrence=sensor_fault_occurrence,
            )
        else:
            template = _select_template(
                seen_anomalies=seen_anomalies,
                mechanism=mechanism,
                sensor_fault_occurrence=1,
            )

        overlay_id = f"UNSEEN_{sequence:03d}"
        host_lot_id = str(
            evaluation_lots.loc[host_index, "lot_id"]
        )

        evaluation_lots, sensor_columns = _copy_template_to_host(
            evaluation_lots=evaluation_lots,
            host_index=host_index,
            template=template,
            mechanism=mechanism,
            overlay_id=overlay_id,
        )

        evidence_ids = [
            f"EVID_UNSEEN_OVERLAY_{overlay_id}"
        ]

        if mechanism in EVENT_BACKED_MECHANISMS:
            event_evidence_id = (
                f"EVID_UNSEEN_EVENT_{overlay_id}"
            )
            event_time = pd.to_datetime(
                evaluation_lots.loc[host_index, "event_time"]
            )
            event_row = _empty_row(
                tool_events_template,
                {
                    "event_id": f"EVT_UNSEEN_{overlay_id}",
                    "event_type": "counterfactual_overlay_alarm",
                    "alarm_code": (
                        "ALARM_UNSEEN_"
                        f"{mechanism.upper()}"
                    ),
                    "tool_id": evaluation_lots.loc[
                        host_index,
                        "tool_id",
                    ],
                    "chamber_id": evaluation_lots.loc[
                        host_index,
                        "chamber_id",
                    ],
                    "start_time": (
                        event_time - pd.Timedelta(hours=1)
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": event_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "severity": "high",
                    "related_lot_id": host_lot_id,
                    "evidence_id": event_evidence_id,
                },
            )
            event_rows.append(event_row)
            evidence_ids.append(event_evidence_id)

        if mechanism == "gradual_degradation":
            maintenance_evidence_id = (
                f"EVID_UNSEEN_MAINT_{overlay_id}"
            )
            event_time = pd.to_datetime(
                evaluation_lots.loc[host_index, "event_time"]
            )
            maintenance_row = _empty_row(
                maintenance_template,
                {
                    "maintenance_id": (
                        f"MAINT_UNSEEN_{overlay_id}"
                    ),
                    "tool_id": evaluation_lots.loc[
                        host_index,
                        "tool_id",
                    ],
                    "chamber_id": evaluation_lots.loc[
                        host_index,
                        "chamber_id",
                    ],
                    "scheduled_at": (
                        event_time - pd.Timedelta(days=5)
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "performed_at": event_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "delay_days": 5,
                    "maintenance_type": (
                        "counterfactual_preventive_inspection"
                    ),
                    "evidence_id": maintenance_evidence_id,
                },
            )
            maintenance_rows.append(maintenance_row)
            evaluation_lots.loc[
                host_index,
                "maintenance_evidence_id",
            ] = maintenance_evidence_id
            evidence_ids.append(maintenance_evidence_id)

        if mechanism == "contextual_anomaly":
            contextual_template = template
            contextual_sensor_columns = sensor_columns
            contextual_overlay_id = overlay_id
            pair_id = "UNSEEN_CONTEXT_PAIR_001"

            evaluation_lots.loc[
                host_index,
                "contextual_pair_id",
            ] = pair_id
            evaluation_lots.loc[
                host_index,
                "contextual_expected_status",
            ] = "anomalous"
            evaluation_lots.loc[
                host_index,
                "is_contextual_control",
            ] = 0
            evaluation_lots.loc[
                host_index,
                "recipe_id",
            ] = "RCP_B"
            evaluation_lots.loc[
                host_index,
                "recipe_context_evidence_id",
            ] = (
                f"EVID_UNSEEN_CONTEXT_ANOMALY_{overlay_id}"
            )
            evidence_ids.append(
                f"EVID_UNSEEN_CONTEXT_ANOMALY_{overlay_id}"
            )

        root_cause_id = str(
            evaluation_lots.loc[host_index, "root_cause_id"]
        )
        rca_row = _empty_row(
            rca_template,
            {
                "case_id": f"RCA_UNSEEN_{overlay_id}",
                "lot_id": host_lot_id,
                "root_cause_id": root_cause_id,
                "suspected_cause": (
                    "Research-only counterfactual transfer of "
                    f"{mechanism} template {template['lot_id']} into "
                    "a held-out TOOL_08 context."
                ),
                "recommended_action": (
                    "Compare against context-matched TOOL_08 behavior "
                    "and review the cited counterfactual evidence."
                ),
                "action": (
                    "Review context-matched history and cited evidence."
                ),
                "outcome": (
                    "Counterfactual unseen-context overlay; not a real "
                    "semiconductor-fab event."
                ),
                "evidence_ids": ";".join(evidence_ids),
                "supports_abstention": False,
            },
        )
        rca_rows.append(rca_row)

        overlay_rows.append(
            {
                "overlay_id": overlay_id,
                "overlay_mode": (
                    "counterfactual_unseen_context_overlay"
                ),
                "host_lot_id": host_lot_id,
                "template_lot_id": template["lot_id"],
                "tool_id": evaluation_lots.loc[
                    host_index,
                    "tool_id",
                ],
                "chamber_id": evaluation_lots.loc[
                    host_index,
                    "chamber_id",
                ],
                "anomaly_mechanism": mechanism,
                "root_cause_id": root_cause_id,
                "transferred_sensor_columns": "|".join(
                    sensor_columns
                ),
            }
        )
        mechanism_counts[mechanism] = (
            mechanism_counts.get(mechanism, 0) + 1
        )

    if contextual_template is None:
        _fail(
            "UNSEEN_TOOL_CHAMBER mechanism_plan must include "
            "contextual_anomaly."
        )

    control_lot_id = str(
        evaluation_lots.loc[contextual_control_index, "lot_id"]
    )

    for column in contextual_sensor_columns:
        contextual_host_index = anomaly_host_indices[
            mechanism_plan.index("contextual_anomaly")
        ]
        evaluation_lots.loc[
            contextual_control_index,
            column,
        ] = evaluation_lots.loc[
            contextual_host_index,
            column,
        ]

    evaluation_lots.loc[
        contextual_control_index,
        "recipe_id",
    ] = "RCP_A"
    evaluation_lots.loc[
        contextual_control_index,
        "is_synthetic_anomaly",
    ] = 0
    evaluation_lots.loc[
        contextual_control_index,
        "anomaly_mechanism",
    ] = "none"
    evaluation_lots.loc[
        contextual_control_index,
        "root_cause_id",
    ] = "none"
    evaluation_lots.loc[
        contextual_control_index,
        "is_contextual_control",
    ] = 1
    evaluation_lots.loc[
        contextual_control_index,
        "contextual_pair_id",
    ] = "UNSEEN_CONTEXT_PAIR_001"
    evaluation_lots.loc[
        contextual_control_index,
        "contextual_expected_status",
    ] = "normal"
    evaluation_lots.loc[
        contextual_control_index,
        "contextual_normal_recipe",
    ] = "RCP_A"
    evaluation_lots.loc[
        contextual_control_index,
        "contextual_anomalous_recipe",
    ] = "RCP_B"
    evaluation_lots.loc[
        contextual_control_index,
        "contextual_sensor_columns",
    ] = "|".join(contextual_sensor_columns)
    evaluation_lots.loc[
        contextual_control_index,
        "recipe_context_evidence_id",
    ] = (
        f"EVID_UNSEEN_CONTEXT_CONTROL_{contextual_overlay_id}"
    )
    evaluation_lots.loc[
        contextual_control_index,
        "synthetic_evidence_id",
    ] = ""
    evaluation_lots.loc[
        contextual_control_index,
        "counterfactual_overlay_id",
    ] = f"{contextual_overlay_id}_CONTROL"
    evaluation_lots.loc[
        contextual_control_index,
        "counterfactual_overlay_mode",
    ] = "counterfactual_unseen_context_overlay"
    evaluation_lots.loc[
        contextual_control_index,
        "counterfactual_template_lot_id",
    ] = contextual_template["lot_id"]

    for rca_row in rca_rows:
        if rca_row.get("case_id") == (
            f"RCA_UNSEEN_{contextual_overlay_id}"
        ):
            rca_row["evidence_ids"] = (
                f"{rca_row['evidence_ids']};"
                f"EVID_UNSEEN_CONTEXT_CONTROL_"
                f"{contextual_overlay_id}"
            )

    evaluation_lots = evaluation_lots.sort_values(
        ["event_time", "lot_id"]
    ).reset_index(drop=True)
    seen_lots = seen_lots.sort_values(
        ["event_time", "lot_id"]
    ).reset_index(drop=True)

    cohort_tables = {
        "train_lots": seen_lots,
        "eval_lots": evaluation_lots,
        "tool_events": pd.DataFrame(
            event_rows,
            columns=tool_events_template.columns,
        ),
        "maintenance": pd.DataFrame(
            maintenance_rows,
            columns=maintenance_template.columns,
        ),
        "process_changes": source_tables[
            "process_changes"
        ].iloc[0:0].copy(),
        "rca_ground_truth": pd.DataFrame(
            rca_rows,
            columns=rca_template.columns,
        ),
        "provenance": pd.DataFrame(overlay_rows),
    }

    return cohort_tables, mechanism_counts, seed


def _known_evidence_ids(
    cohort_tables: dict[str, pd.DataFrame],
) -> set[str]:
    known_ids: set[str] = set()

    for table_id, column in (
        ("tool_events", "evidence_id"),
        ("maintenance", "evidence_id"),
        ("process_changes", "evidence_id"),
        ("eval_lots", "synthetic_evidence_id"),
        ("eval_lots", "maintenance_evidence_id"),
        ("eval_lots", "recipe_context_evidence_id"),
    ):
        frame = cohort_tables[table_id]

        if column in frame.columns:
            known_ids.update(
                frame.loc[_non_empty(frame[column]), column]
                .astype(str)
                .str.strip()
            )

    return known_ids


def _validate_cohort_tables(
    cohort_tables: dict[str, pd.DataFrame],
    scenario: dict[str, Any],
) -> UnseenContextSummary:
    contract = _contract(scenario)
    train_lots = cohort_tables["train_lots"]
    eval_lots = cohort_tables["eval_lots"]
    rca_ground_truth = cohort_tables["rca_ground_truth"]
    provenance = cohort_tables["provenance"]

    if _true_mask(train_lots["is_unseen_context"]).any():
        _fail("Training split must contain seen contexts only.")

    if not _true_mask(eval_lots["is_unseen_context"]).all():
        _fail("Evaluation split must contain unseen contexts only.")

    if set(eval_lots["tool_id"].astype(str)) != {"TOOL_08"}:
        _fail("Evaluation split must use held-out TOOL_08 only.")

    anomaly_mask = _true_mask(eval_lots["is_synthetic_anomaly"])
    anomaly_lots = eval_lots.loc[anomaly_mask].copy()
    expected_anomaly_count = int(
        contract["target_anomaly_count"]
    )

    if len(anomaly_lots) != expected_anomaly_count:
        _fail(
            "Unexpected unseen-context synthetic anomaly count."
        )

    achieved_anomaly_rate = len(anomaly_lots) / len(eval_lots)
    expected_anomaly_rate = float(
        contract["achieved_anomaly_rate"]
    )

    if abs(achieved_anomaly_rate - expected_anomaly_rate) > 1e-9:
        _fail(
            "Unexpected unseen-context achieved anomaly prevalence. "
            f"Expected {expected_anomaly_rate:.10f}, "
            f"got {achieved_anomaly_rate:.10f}."
        )

    expected_mechanism_counts: dict[str, int] = {}

    for mechanism in contract["mechanism_plan"]:
        expected_mechanism_counts[mechanism] = (
            expected_mechanism_counts.get(mechanism, 0) + 1
        )

    observed_mechanism_counts = {
        mechanism: int(
            anomaly_lots["anomaly_mechanism"]
            .eq(mechanism)
            .sum()
        )
        for mechanism in expected_mechanism_counts
    }

    if observed_mechanism_counts != expected_mechanism_counts:
        _fail(
            "Unexpected unseen-context anomaly mechanism distribution."
        )

    if len(provenance) != expected_anomaly_count:
        _fail(
            "Overlay provenance must contain one row per anomaly."
        )

    if not _non_empty(
        anomaly_lots["counterfactual_overlay_id"]
    ).all():
        _fail("Every overlay anomaly requires an overlay ID.")

    contextual_anomaly = anomaly_lots.loc[
        anomaly_lots["anomaly_mechanism"].eq(
            "contextual_anomaly"
        )
    ]

    if len(contextual_anomaly) != 1:
        _fail(
            "Unseen overlay must contain one contextual anomaly."
        )

    pair_id = str(
        contextual_anomaly.iloc[0]["contextual_pair_id"]
    )
    contextual_control = eval_lots.loc[
        _true_mask(eval_lots["is_contextual_control"])
        & eval_lots["contextual_pair_id"].astype(str).eq(pair_id)
    ]

    if len(contextual_control) != 1:
        _fail(
            "Unseen contextual anomaly requires one Recipe A control."
        )

    sensor_columns = _parse_sensor_columns(
        contextual_anomaly.iloc[0]["contextual_sensor_columns"]
    )

    anomaly_values = pd.to_numeric(
        contextual_anomaly.iloc[0][sensor_columns],
        errors="coerce",
    )
    control_values = pd.to_numeric(
        contextual_control.iloc[0][sensor_columns],
        errors="coerce",
    )

    if not anomaly_values.equals(control_values):
        _fail(
            "Unseen contextual anomaly and control must share one "
            "sensor profile."
        )

    anomaly_lot_ids = set(anomaly_lots["lot_id"].astype(str))

    if set(rca_ground_truth["lot_id"].astype(str)) != anomaly_lot_ids:
        _fail(
            "RCA ground truth must cover exactly unseen overlay anomalies."
        )

    known_evidence_ids = _known_evidence_ids(cohort_tables)

    for case_id, evidence_ids in zip(
        rca_ground_truth["case_id"].astype(str),
        rca_ground_truth["evidence_ids"],
    ):
        missing_evidence = _evidence_tokens(
            evidence_ids
        ).difference(known_evidence_ids)

        if missing_evidence:
            _fail(
                f"RCA case {case_id} references unavailable evidence: "
                f"{sorted(missing_evidence)}"
            )

    expected_event_count = sum(
        count
        for mechanism, count in expected_mechanism_counts.items()
        if mechanism in EVENT_BACKED_MECHANISMS
    )

    if len(cohort_tables["tool_events"]) != expected_event_count:
        _fail(
            "Unexpected unseen overlay tool-event count."
        )

    if len(cohort_tables["maintenance"]) != 1:
        _fail(
            "Gradual degradation overlay requires one maintenance record."
        )

    return UnseenContextSummary(
        training_lot_count=len(train_lots),
        evaluation_lot_count=len(eval_lots),
        synthetic_anomaly_count=len(anomaly_lots),
        achieved_anomaly_rate=achieved_anomaly_rate,
        mechanism_counts=observed_mechanism_counts,
        output_dir=Path(),
    )


def _write_outputs(
    cohort_tables: dict[str, pd.DataFrame],
    output_dir: Path,
    manifest_payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    for table_id, filename in OUTPUT_FILES.items():
        if table_id == "manifest":
            continue

        path = output_dir / filename
        cohort_tables[table_id].to_csv(
            path,
            index=False,
            lineterminator="\n",
        )
        hashes[filename] = _sha256(path)

    manifest_payload["output_sha256"] = hashes
    (output_dir / OUTPUT_FILES["manifest"]).write_text(
        json.dumps(
            manifest_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_unseen_tool_chamber(
    repo_root: Path,
    output_dir: Path | None = None,
) -> UnseenContextSummary:
    """Materialize and validate the held-out TOOL_08 overlay."""
    repo_root = repo_root.resolve()
    manifest, scenario = _load_scenario(repo_root)
    source_tables = _load_tables(repo_root)

    cohort_tables, mechanism_counts, seed = _build_overlay(
        source_tables=source_tables,
        manifest=manifest,
        scenario=scenario,
    )

    if output_dir is None:
        output_dir = (
            repo_root
            / "data"
            / "synthetic"
            / "v2"
            / "scenarios"
            / "UNSEEN_TOOL_CHAMBER"
        )

    output_dir = output_dir.resolve()
    summary = _validate_cohort_tables(
        cohort_tables=cohort_tables,
        scenario=scenario,
    )

    manifest_payload = {
        "schema_version": "1.0",
        "scenario_id": "UNSEEN_TOOL_CHAMBER",
        "overlay_mode": (
            "counterfactual_unseen_context_overlay"
        ),
        "disclaimer": scenario["materialization"]["disclaimer"],
        "random_seed": seed,
        "training_lot_count": summary.training_lot_count,
        "evaluation_lot_count": summary.evaluation_lot_count,
        "synthetic_anomaly_count": summary.synthetic_anomaly_count,
        "achieved_synthetic_anomaly_rate": (
            summary.achieved_anomaly_rate
        ),
        "mechanism_counts": mechanism_counts,
        "source_lots": BASE_OUTPUT_FILES["lots"],
        "preserves_core_v2_outputs": True,
    }

    _write_outputs(
        cohort_tables=cohort_tables,
        output_dir=output_dir,
        manifest_payload=manifest_payload,
    )

    summary = UnseenContextSummary(
        training_lot_count=summary.training_lot_count,
        evaluation_lot_count=summary.evaluation_lot_count,
        synthetic_anomaly_count=summary.synthetic_anomaly_count,
        achieved_anomaly_rate=summary.achieved_anomaly_rate,
        mechanism_counts=summary.mechanism_counts,
        output_dir=output_dir,
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the Synthetic Data V2 UNSEEN_TOOL_CHAMBER "
            "counterfactual overlay."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "data/synthetic/v2/scenarios/UNSEEN_TOOL_CHAMBER."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        summary = materialize_unseen_tool_chamber(
            repo_root=Path.cwd(),
            output_dir=args.output_dir,
        )
    except (
        OSError,
        json.JSONDecodeError,
        UnseenContextMaterializationError,
    ) as error:
        print("UNSEEN_TOOL_CHAMBER_MATERIALIZATION_FAILED")
        print(f"- {error}")
        return 1

    mechanism_text = ", ".join(
        f"{mechanism}={count}"
        for mechanism, count in summary.mechanism_counts.items()
    )

    print("UNSEEN_TOOL_CHAMBER_MATERIALIZATION_OK")
    print(
        "Overlay mode: counterfactual_unseen_context_overlay"
    )
    print(f"Training lots: {summary.training_lot_count}")
    print(f"Evaluation lots: {summary.evaluation_lot_count}")
    print(
        "Evaluation synthetic anomaly prevalence: "
        f"{summary.achieved_anomaly_rate:.2%}"
    )
    print(
        "Evaluation synthetic anomaly lots: "
        f"{summary.synthetic_anomaly_count}"
    )
    print(f"Mechanism counts: {mechanism_text}")
    print("Held-out tool: TOOL_08")
    print(f"Output directory: {summary.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())