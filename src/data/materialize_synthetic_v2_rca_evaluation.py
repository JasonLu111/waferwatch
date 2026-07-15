"""Materialize supported Synthetic Data V2 RCA evaluation cohorts.

This module currently supports all five manifest RCA scenarios:
``RCA_ABRUPT_MEAN_SHIFT``, ``RCA_GRADUAL_DEGRADATION``,
``RCA_VARIANCE_INSTABILITY``, ``RCA_SENSOR_FAULT``, and
``RCA_CONTEXTUAL_ANOMALY``.  Each cohort is a deterministic,
evidence-complete subset of the core Synthetic Data V2 outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCENARIO_ID = "RCA_ABRUPT_MEAN_SHIFT"

SCENARIO_SPECS: dict[str, dict[str, str]] = {
    "RCA_ABRUPT_MEAN_SHIFT": {
        "root_cause_id": "RC_PRESSURE_INSTABILITY",
        "anomaly_mechanism": "abrupt_mean_shift",
        "context_evidence_type": "tool_event",
        "source_path_key": "source_tool_events_path",
        "expected_count_key": "expected_tool_event_count",
        "required_evidence_id_key": "required_tool_event_evidence_id",
        "source_table": "synthetic_tool_events",
        "output_key": "tool_events",
    },
    "RCA_GRADUAL_DEGRADATION": {
        "root_cause_id": "RC_COMPONENT_WEAR",
        "anomaly_mechanism": "gradual_degradation",
        "context_evidence_type": "maintenance",
        "source_path_key": "source_maintenance_path",
        "expected_count_key": "expected_maintenance_count",
        "required_evidence_id_key": "required_maintenance_evidence_id",
        "source_table": "synthetic_maintenance",
        "output_key": "maintenance",
    },
    "RCA_VARIANCE_INSTABILITY": {
        "root_cause_id": "RC_PROCESS_VARIABILITY",
        "anomaly_mechanism": "variance_instability",
        "context_evidence_type": "tool_event",
        "source_path_key": "source_tool_events_path",
        "expected_count_key": "expected_tool_event_count",
        "required_evidence_id_key": "required_tool_event_evidence_id",
        "source_table": "synthetic_tool_events",
        "output_key": "tool_events",
    },
    "RCA_SENSOR_FAULT": {
        "root_cause_id": "RC_SENSOR_HARDWARE",
        "anomaly_mechanism": "sensor_fault",
        "context_evidence_type": "tool_event",
        "source_path_key": "source_tool_events_path",
        "expected_count_key": "expected_tool_event_count",
        "required_evidence_id_key": "required_tool_event_evidence_ids",
        "source_table": "synthetic_tool_events",
        "output_key": "tool_events",
    },
    "RCA_CONTEXTUAL_ANOMALY": {
        "root_cause_id": "RC_RECIPE_CONTEXT_MISMATCH",
        "anomaly_mechanism": "contextual_anomaly",
        "context_evidence_type": "tool_event",
        "source_table": "synthetic_tool_events",
        "output_key": "tool_events",
    },
}

SCENARIO_OUTPUT_FILES: dict[str, dict[str, str]] = {
    "RCA_ABRUPT_MEAN_SHIFT": {
        "lots": "rca_abrupt_mean_shift_lots.csv",
        "ground_truth": "rca_abrupt_mean_shift_ground_truth.csv",
        "tool_events": "rca_abrupt_mean_shift_tool_events.csv",
        "evidence_bundle": "rca_abrupt_mean_shift_evidence_bundle.csv",
        "manifest": "rca_abrupt_mean_shift_cohort_manifest.json",
    },
    "RCA_GRADUAL_DEGRADATION": {
        "lots": "rca_gradual_degradation_lots.csv",
        "ground_truth": "rca_gradual_degradation_ground_truth.csv",
        "maintenance": "rca_gradual_degradation_maintenance.csv",
        "evidence_bundle": "rca_gradual_degradation_evidence_bundle.csv",
        "manifest": "rca_gradual_degradation_cohort_manifest.json",
    },
    "RCA_VARIANCE_INSTABILITY": {
        "lots": "rca_variance_instability_lots.csv",
        "ground_truth": "rca_variance_instability_ground_truth.csv",
        "tool_events": "rca_variance_instability_tool_events.csv",
        "evidence_bundle": "rca_variance_instability_evidence_bundle.csv",
        "manifest": "rca_variance_instability_cohort_manifest.json",
    },
    "RCA_SENSOR_FAULT": {
        "lots": "rca_sensor_fault_lots.csv",
        "ground_truth": "rca_sensor_fault_ground_truth.csv",
        "tool_events": "rca_sensor_fault_tool_events.csv",
        "evidence_bundle": "rca_sensor_fault_evidence_bundle.csv",
        "manifest": "rca_sensor_fault_cohort_manifest.json",
    },
    "RCA_CONTEXTUAL_ANOMALY": {
        "lots": "rca_contextual_anomaly_lots.csv",
        "ground_truth": "rca_contextual_anomaly_ground_truth.csv",
        "tool_events": "rca_contextual_anomaly_tool_events.csv",
        "recipe_context": "rca_contextual_anomaly_recipe_context.csv",
        "evidence_bundle": "rca_contextual_anomaly_evidence_bundle.csv",
        "manifest": "rca_contextual_anomaly_cohort_manifest.json",
    },
}

RCA_ABRUPT_MEAN_SHIFT_OUTPUT_FILES = SCENARIO_OUTPUT_FILES[
    "RCA_ABRUPT_MEAN_SHIFT"
]
RCA_GRADUAL_DEGRADATION_OUTPUT_FILES = SCENARIO_OUTPUT_FILES[
    "RCA_GRADUAL_DEGRADATION"
]
RCA_VARIANCE_INSTABILITY_OUTPUT_FILES = SCENARIO_OUTPUT_FILES[
    "RCA_VARIANCE_INSTABILITY"
]
RCA_SENSOR_FAULT_OUTPUT_FILES = SCENARIO_OUTPUT_FILES["RCA_SENSOR_FAULT"]
RCA_CONTEXTUAL_ANOMALY_OUTPUT_FILES = SCENARIO_OUTPUT_FILES[
    "RCA_CONTEXTUAL_ANOMALY"
]

EVIDENCE_BUNDLE_COLUMNS = [
    "evidence_id",
    "evidence_type",
    "source_table",
    "source_record_id",
    "lot_id",
    "event_time",
    "tool_id",
    "chamber_id",
    "root_cause_id",
    "is_shared_evidence",
    "evidence_summary",
    "evidence_payload",
]


class RcaEvaluationMaterializationError(RuntimeError):
    """Raised when an RCA cohort contract cannot be materialized safely."""


@dataclass(frozen=True)
class RcaEvaluationSummary:
    scenario_id: str
    root_cause_id: str
    cohort_lot_count: int
    ground_truth_count: int
    tool_event_count: int
    maintenance_count: int
    context_evidence_type: str
    context_evidence_count: int
    recipe_context_count: int
    evidence_bundle_count: int
    output_dir: Path


def _fail(message: str) -> None:
    raise RcaEvaluationMaterializationError(message)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _true_mask(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true"})


def _split_evidence_ids(value: Any) -> set[str]:
    return {
        token.strip()
        for token in re.split(r"[,;|]", _as_text(value))
        if token.strip()
    }


def _require_columns(
    table: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        _fail(
            f"{table_name} is missing required columns: {', '.join(missing)}."
        )


def _sort_table(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available_columns = [column for column in columns if column in table.columns]
    if not available_columns:
        return table.reset_index(drop=True)
    return table.sort_values(available_columns, kind="stable").reset_index(
        drop=True
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        _fail(f"Required file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        _fail(f"Invalid JSON in {path}: {error}")
    if not isinstance(payload, dict):
        _fail(f"Expected a JSON object in {path}.")
    return payload


def _resolve_repo_path(repo_root: Path, configured_path: Any) -> Path:
    candidate = Path(_as_text(configured_path))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        _fail(f"Configured source path escapes the repository: {configured_path}")
    return resolved


def _scenario_from_manifest(
    manifest: dict[str, Any],
    scenario_id: str,
) -> dict[str, Any]:
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list):
        _fail("Stress-test manifest must contain a scenarios list.")
    matches = [
        scenario
        for scenario in scenarios
        if isinstance(scenario, dict) and scenario.get("id") == scenario_id
    ]
    if len(matches) != 1:
        _fail(
            f"Expected exactly one manifest scenario named {scenario_id}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _materialization_contract(
    scenario_id: str,
    scenario: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    spec = SCENARIO_SPECS.get(scenario_id)
    if spec is None:
        _fail(f"Unsupported RCA scenario: {scenario_id}.")

    context_evidence_type = spec["context_evidence_type"]
    if scenario.get("scenario_type") != "root_cause":
        _fail(f"{scenario_id} must have scenario_type root_cause.")
    if scenario.get("root_cause_id") != spec["root_cause_id"]:
        _fail(
            f"{scenario_id} must target {spec['root_cause_id']}."
        )
    if set(scenario.get("required_evidence_types", [])) != {
        context_evidence_type,
        "synthetic_lot",
    }:
        _fail(
            f"{scenario_id} must require {context_evidence_type} and "
            "synthetic_lot evidence."
        )
    if tuple(scenario.get("expected_top_k", [])) != (1, 3):
        _fail(f"{scenario_id} must declare expected_top_k [1, 3].")
    if scenario.get("expected_abstention") is not False:
        _fail(f"{scenario_id} must not allow abstention.")

    raw_contract = scenario.get("materialization")
    if raw_contract is not None and not isinstance(raw_contract, dict):
        _fail(f"{scenario_id} materialization must be a JSON object.")

    # R5.22 preceded per-scenario materialization metadata.  Keep its public
    # abrupt cohort function and its compact fixture contract compatible while
    # requiring explicit metadata for new RCA checkpoints.
    if scenario_id != "RCA_ABRUPT_MEAN_SHIFT" and not isinstance(
        raw_contract, dict
    ):
        _fail(
            f"{scenario_id} requires a materialization object in the "
            "stress-test manifest."
        )

    default_source_path = {
        "tool_event": "data/synthetic/v2/synthetic_tool_events.csv",
        "maintenance": "data/synthetic/v2/synthetic_maintenance.csv",
    }[context_evidence_type]
    contract: dict[str, Any] = {
        "mode": "rca_evaluation_cohort",
        "source_lots_path": "data/synthetic/v2/synthetic_secom_v2.csv",
        "source_rca_ground_truth_path": (
            "data/synthetic/v2/synthetic_rca_ground_truth.csv"
        ),
        spec["source_path_key"]: default_source_path,
        "preserve_core_v2_outputs": True,
        "disclaimer": (
            "Research-only Synthetic Data V2 RCA evaluation cohort. "
            "It is not data from a real semiconductor fab."
        ),
    }
    if isinstance(raw_contract, dict):
        contract.update(raw_contract)

    required_fields = [
        "mode",
        "source_lots_path",
        "source_rca_ground_truth_path",
        spec["source_path_key"],
        "expected_lot_count",
        "expected_ground_truth_count",
        spec["expected_count_key"],
        "expected_evidence_bundle_count",
        spec["required_evidence_id_key"],
        "preserve_core_v2_outputs",
        "disclaimer",
    ]
    # Only new scenarios must declare a fully explicit materialization
    # contract.  The R5.22 abrupt scenario deliberately retains legacy
    # fixture compatibility and derives any omitted count from its source.
    if scenario_id == "RCA_ABRUPT_MEAN_SHIFT":
        required_fields = [
            "mode",
            "source_lots_path",
            "source_rca_ground_truth_path",
            spec["source_path_key"],
            "preserve_core_v2_outputs",
            "disclaimer",
        ]
    missing = [field for field in required_fields if field not in contract]
    if missing:
        _fail(
            f"{scenario_id} materialization is missing fields: "
            f"{', '.join(missing)}."
        )
    if contract["mode"] != "rca_evaluation_cohort":
        _fail("RCA materialization mode must be rca_evaluation_cohort.")
    if contract["preserve_core_v2_outputs"] is not True:
        _fail("RCA materialization must preserve core V2 outputs.")
    if not _as_text(contract["disclaimer"]):
        _fail("RCA materialization must include a research-only disclaimer.")
    return contract, spec


def _contract_evidence_ids(
    value: Any,
) -> set[str]:
    """Normalize a singular evidence ID or a manifest list of IDs."""
    if isinstance(value, list):
        evidence_ids = {_as_text(item).strip() for item in value}
        if "" in evidence_ids or not evidence_ids:
            _fail("Required evidence IDs must contain non-empty values.")
        if len(evidence_ids) != len(value):
            _fail("Required evidence IDs must not contain duplicates.")
        return evidence_ids
    return _split_evidence_ids(value)


def _first_present(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        if column in row.index:
            value = _as_text(row[column])
            if value:
                return value
    return ""


def _json_payload(row: pd.Series, columns: list[str]) -> str:
    payload = {
        column: _as_text(row[column])
        for column in columns
        if column in row.index
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_and_select_tables(
    lots: pd.DataFrame,
    context_evidence: pd.DataFrame,
    ground_truth: pd.DataFrame,
    scenario: dict[str, Any],
    contract: dict[str, Any],
    spec: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    scenario_id = _as_text(scenario["id"])
    root_cause_id = _as_text(scenario["root_cause_id"])
    context_type = spec["context_evidence_type"]
    _require_columns(
        lots,
        [
            "lot_id",
            "event_time",
            "tool_id",
            "chamber_id",
            "anomaly_mechanism",
            "root_cause_id",
            "is_synthetic_anomaly",
            "synthetic_evidence_id",
        ],
        "Synthetic V2 lots",
    )
    _require_columns(
        ground_truth,
        [
            "case_id",
            "lot_id",
            "root_cause_id",
            "evidence_ids",
            "supports_abstention",
            "top3_acceptable_causes",
        ],
        "Synthetic V2 RCA ground truth",
    )
    _require_columns(
        context_evidence,
        ["evidence_id"],
        f"Synthetic V2 {context_type} evidence",
    )

    selected_truth = ground_truth.loc[
        ground_truth["root_cause_id"].astype(str).eq(root_cause_id)
    ].copy()
    selected_truth = _sort_table(selected_truth, ["case_id", "lot_id"])
    expected_truth_count = contract.get("expected_ground_truth_count")
    if (
        expected_truth_count is not None
        and len(selected_truth) != int(expected_truth_count)
    ):
        _fail(
            "Unexpected RCA ground-truth count: "
            f"expected {expected_truth_count}, got {len(selected_truth)}."
        )
    if selected_truth["lot_id"].astype(str).duplicated().any():
        _fail(f"{scenario_id} ground truth must contain one case per lot.")
    if _true_mask(selected_truth["supports_abstention"]).any():
        _fail(f"{scenario_id} ground truth must not support abstention.")

    selected_lot_ids = set(selected_truth["lot_id"].astype(str))
    selected_lots = lots.loc[
        lots["lot_id"].astype(str).isin(selected_lot_ids)
    ].copy()
    selected_lots = _sort_table(selected_lots, ["event_time", "lot_id"])
    expected_lot_count = contract.get("expected_lot_count")
    if expected_lot_count is not None and len(selected_lots) != int(
        expected_lot_count
    ):
        _fail(
            "Unexpected RCA cohort lot count: "
            f"expected {expected_lot_count}, got {len(selected_lots)}."
        )
    if set(selected_lots["lot_id"].astype(str)) != selected_lot_ids:
        _fail("RCA ground-truth lot IDs do not match the selected cohort lots.")
    if set(selected_lots["root_cause_id"].astype(str)) != {root_cause_id}:
        _fail("Selected RCA lots have an unexpected root-cause ID.")
    if set(selected_lots["anomaly_mechanism"].astype(str)) != {
        spec["anomaly_mechanism"]
    }:
        _fail(
            f"Selected RCA lots must be {spec['anomaly_mechanism']} "
            "anomalies only."
        )
    if not _true_mask(selected_lots["is_synthetic_anomaly"]).all():
        _fail("Selected RCA lots must all be synthetic anomalies.")

    truth_evidence_by_lot = {
        _as_text(row.lot_id): _split_evidence_ids(row.evidence_ids)
        for row in selected_truth.itertuples(index=False)
    }
    truth_evidence_ids = set().union(*truth_evidence_by_lot.values())
    lot_evidence_ids = set(selected_lots["synthetic_evidence_id"].map(_as_text))
    if "" in lot_evidence_ids:
        _fail("Selected RCA lots must each cite a synthetic evidence ID.")
    for row in selected_lots.itertuples(index=False):
        lot_id = _as_text(row.lot_id)
        lot_evidence_id = _as_text(row.synthetic_evidence_id)
        if lot_evidence_id not in truth_evidence_by_lot[lot_id]:
            _fail(
                f"RCA case for {lot_id} does not cite its synthetic lot evidence ID."
            )

    selected_context_evidence = context_evidence.loc[
        context_evidence["evidence_id"].astype(str).isin(truth_evidence_ids)
    ].copy()
    selected_context_evidence = _sort_table(
        selected_context_evidence,
        [
            "performed_at",
            "event_time",
            "occurred_at",
            "start_time",
            "end_time",
            "scheduled_at",
            "evidence_id",
        ],
    )
    expected_context_count = contract.get(spec["expected_count_key"])
    if (
        expected_context_count is not None
        and len(selected_context_evidence) != int(expected_context_count)
    ):
        _fail(
            f"Unexpected RCA {context_type} evidence count: expected "
            f"{expected_context_count}, got {len(selected_context_evidence)}."
        )
    required_context_evidence_ids = _contract_evidence_ids(
        contract.get(spec["required_evidence_id_key"])
    )
    if (
        required_context_evidence_ids
        and set(selected_context_evidence["evidence_id"].astype(str))
        != required_context_evidence_ids
    ):
        _fail(
            f"RCA {context_type} evidence does not match the manifest contract."
        )

    bundle_evidence_ids = lot_evidence_ids | set(
        selected_context_evidence["evidence_id"].astype(str)
    )
    if bundle_evidence_ids != truth_evidence_ids:
        _fail(
            "RCA evidence bundle would not cover every ground-truth evidence ID."
        )
    expected_bundle_count = contract.get("expected_evidence_bundle_count")
    if (
        expected_bundle_count is not None
        and len(bundle_evidence_ids) != int(expected_bundle_count)
    ):
        _fail(
            "Unexpected RCA evidence-bundle count: "
            f"expected {expected_bundle_count}, got {len(bundle_evidence_ids)}."
        )
    return (
        selected_lots,
        selected_truth,
        selected_context_evidence,
        bundle_evidence_ids,
    )


def _context_evidence_record(
    row: pd.Series,
    root_cause_id: str,
    spec: dict[str, str],
) -> dict[str, Any]:
    context_type = spec["context_evidence_type"]
    if context_type == "maintenance":
        summary = (
            "Synthetic delayed-maintenance evidence; "
            f"type={_first_present(row, ['maintenance_type'])}; "
            f"component={_first_present(row, ['component'])}; "
            f"delay_days={_first_present(row, ['delay_days'])}; "
            f"replacement_performed={_first_present(row, ['replacement_performed'])}."
        )
        source_record_id = _first_present(
            row, ["maintenance_id", "evidence_id"]
        )
        event_time = _first_present(row, ["performed_at", "scheduled_at"])
        lot_id = _first_present(row, ["related_lot_id", "lot_id"])
    else:
        event_type = _first_present(row, ["event_type", "event_name", "event_category"])
        description = _first_present(row, ["description", "event_description", "message"])
        summary = "; ".join(value for value in [event_type, description] if value)
        source_record_id = _first_present(
            row, ["event_id", "tool_event_id", "evidence_id"]
        )
        event_time = _first_present(
            row, ["event_time", "occurred_at", "start_time", "end_time"]
        )
        lot_id = _first_present(row, ["related_lot_id", "lot_id"])

    return {
        "_sort_order": 0,
        "evidence_id": _as_text(row["evidence_id"]),
        "evidence_type": context_type,
        "source_table": spec["source_table"],
        "source_record_id": source_record_id,
        "lot_id": lot_id,
        "event_time": event_time,
        "tool_id": _first_present(row, ["tool_id"]),
        "chamber_id": _first_present(row, ["chamber_id"]),
        "root_cause_id": root_cause_id,
        "is_shared_evidence": True,
        "evidence_summary": summary or f"Synthetic {context_type} evidence.",
        "evidence_payload": _json_payload(row, list(row.index)),
    }


def _build_evidence_bundle(
    lots: pd.DataFrame,
    context_evidence: pd.DataFrame,
    root_cause_id: str,
    spec: dict[str, str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for context_row in context_evidence.itertuples(index=False):
        records.append(
            _context_evidence_record(
                pd.Series(context_row._asdict()), root_cause_id, spec
            )
        )

    lot_payload_columns = [
        "lot_id",
        "event_time",
        "tool_id",
        "chamber_id",
        "recipe_id",
        "product_family",
        "anomaly_mechanism",
        "root_cause_id",
        "injected_sensor_columns",
        "degradation_progress",
        "degradation_window_position",
        "maximum_magnitude_sigma",
        "abrupt_shift_sigma",
        "synthetic_evidence_id",
    ]
    for lot in lots.itertuples(index=False):
        row = pd.Series(lot._asdict())
        mechanism = _as_text(row["anomaly_mechanism"])
        records.append(
            {
                "_sort_order": 1,
                "evidence_id": _as_text(row["synthetic_evidence_id"]),
                "evidence_type": "synthetic_lot",
                "source_table": "synthetic_secom_v2",
                "source_record_id": _as_text(row["lot_id"]),
                "lot_id": _as_text(row["lot_id"]),
                "event_time": _as_text(row["event_time"]),
                "tool_id": _as_text(row["tool_id"]),
                "chamber_id": _as_text(row["chamber_id"]),
                "root_cause_id": root_cause_id,
                "is_shared_evidence": False,
                "evidence_summary": f"Synthetic {mechanism} lot evidence.",
                "evidence_payload": _json_payload(row, lot_payload_columns),
            }
        )

    bundle = pd.DataFrame(records)
    bundle = bundle.sort_values(
        ["_sort_order", "evidence_id"], kind="stable"
    ).reset_index(drop=True)
    return bundle.drop(columns="_sort_order")[EVIDENCE_BUNDLE_COLUMNS]


CONTEXTUAL_RECIPE_CONTEXT_COLUMNS = [
    "lot_id",
    "event_time",
    "tool_id",
    "chamber_id",
    "recipe_id",
    "product_family",
    "is_synthetic_anomaly",
    "anomaly_mechanism",
    "root_cause_id",
    "synthetic_evidence_id",
    "is_contextual_control",
    "contextual_pair_id",
    "contextual_expected_status",
    "contextual_normal_recipe",
    "contextual_anomalous_recipe",
    "contextual_sensor_columns",
    "recipe_context_evidence_id",
]


def _contextual_contract(scenario: dict[str, Any]) -> dict[str, Any]:
    """Validate the explicit contract for paired contextual RCA evidence."""
    scenario_id = "RCA_CONTEXTUAL_ANOMALY"
    if scenario.get("scenario_type") != "root_cause":
        _fail(f"{scenario_id} must have scenario_type root_cause.")
    if scenario.get("root_cause_id") != "RC_RECIPE_CONTEXT_MISMATCH":
        _fail(
            f"{scenario_id} must target RC_RECIPE_CONTEXT_MISMATCH."
        )
    if set(scenario.get("required_evidence_types", [])) != {
        "tool_event",
        "recipe_context",
        "synthetic_lot",
    }:
        _fail(
            f"{scenario_id} must require tool_event, recipe_context, and "
            "synthetic_lot evidence."
        )
    if tuple(scenario.get("expected_top_k", [])) != (1, 3):
        _fail(f"{scenario_id} must declare expected_top_k [1, 3].")
    if scenario.get("expected_abstention") is not False:
        _fail(f"{scenario_id} must not allow abstention.")

    contract = scenario.get("materialization")
    if not isinstance(contract, dict):
        _fail(
            f"{scenario_id} requires a materialization object in the "
            "stress-test manifest."
        )

    required_fields = [
        "mode",
        "source_lots_path",
        "source_tool_events_path",
        "source_rca_ground_truth_path",
        "expected_lot_count",
        "expected_ground_truth_count",
        "expected_tool_event_count",
        "expected_recipe_context_count",
        "expected_evidence_bundle_count",
        "required_tool_event_evidence_id",
        "preserve_core_v2_outputs",
        "disclaimer",
    ]
    missing = [field for field in required_fields if field not in contract]
    if missing:
        _fail(
            f"{scenario_id} materialization is missing fields: "
            f"{', '.join(missing)}."
        )
    if contract["mode"] != "rca_evaluation_cohort":
        _fail("RCA materialization mode must be rca_evaluation_cohort.")
    if contract["preserve_core_v2_outputs"] is not True:
        _fail("RCA materialization must preserve core V2 outputs.")
    if not _as_text(contract["disclaimer"]):
        _fail("RCA materialization must include a research-only disclaimer.")
    return contract


def _build_contextual_recipe_context(
    lots: pd.DataFrame,
    anomaly_lots: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    _require_columns(
        lots,
        CONTEXTUAL_RECIPE_CONTEXT_COLUMNS,
        "Synthetic V2 contextual lots",
    )
    if _true_mask(anomaly_lots["is_contextual_control"]).any():
        _fail("Contextual RCA anomaly lots must not be marked as controls.")

    pair_ids = anomaly_lots["contextual_pair_id"].map(_as_text)
    if "" in set(pair_ids):
        _fail("Contextual RCA anomaly lots must each declare a pair ID.")
    if pair_ids.duplicated().any():
        _fail("Contextual RCA anomaly lots must contain one anomaly per pair.")

    selected_pair_ids = set(pair_ids)
    pair_rows = lots.loc[
        lots["contextual_pair_id"].astype(str).isin(selected_pair_ids)
    ].copy()
    controls = pair_rows.loc[_true_mask(pair_rows["is_contextual_control"])].copy()
    controls = _sort_table(controls, ["contextual_pair_id", "event_time", "lot_id"])
    if len(controls) != len(anomaly_lots):
        _fail(
            "Contextual RCA must contain exactly one normal control per "
            "anomaly pair."
        )
    if set(controls["contextual_pair_id"].map(_as_text)) != selected_pair_ids:
        _fail("Contextual RCA normal controls do not cover every anomaly pair.")
    if _true_mask(controls["is_synthetic_anomaly"]).any():
        _fail("Contextual RCA normal controls must not be synthetic anomalies.")

    recipe_context = pd.concat([anomaly_lots, controls], ignore_index=True)
    recipe_context = _sort_table(
        recipe_context,
        ["contextual_pair_id", "is_contextual_control", "event_time", "lot_id"],
    )
    expected_count = int(contract["expected_recipe_context_count"])
    if len(recipe_context) != expected_count:
        _fail(
            "Unexpected contextual recipe-context evidence count: expected "
            f"{expected_count}, got {len(recipe_context)}."
        )

    for pair_id in sorted(selected_pair_ids):
        pair = recipe_context.loc[
            recipe_context["contextual_pair_id"].astype(str).eq(pair_id)
        ]
        pair_controls = pair.loc[_true_mask(pair["is_contextual_control"])]
        pair_anomalies = pair.loc[~_true_mask(pair["is_contextual_control"])]
        if len(pair_controls) != 1 or len(pair_anomalies) != 1:
            _fail(
                f"Contextual pair {pair_id} must contain one control and "
                "one anomaly."
            )
        control = pair_controls.iloc[0]
        anomaly = pair_anomalies.iloc[0]
        for row in [control, anomaly]:
            if not _as_text(row["contextual_expected_status"]):
                _fail(
                    f"Contextual pair {pair_id} has a missing expected status."
                )
            if not _as_text(row["contextual_sensor_columns"]):
                _fail(
                    f"Contextual pair {pair_id} has missing sensor context."
                )
            if not _as_text(row["recipe_context_evidence_id"]):
                _fail(
                    f"Contextual pair {pair_id} has missing recipe-context evidence."
                )
        if _as_text(control["recipe_id"]) != _as_text(
            control["contextual_normal_recipe"]
        ):
            _fail(f"Contextual control for {pair_id} must use the normal recipe.")
        if _as_text(anomaly["recipe_id"]) != _as_text(
            anomaly["contextual_anomalous_recipe"]
        ):
            _fail(f"Contextual anomaly for {pair_id} must use the anomalous recipe.")
        if _as_text(control["contextual_normal_recipe"]) != _as_text(
            anomaly["contextual_normal_recipe"]
        ) or _as_text(control["contextual_anomalous_recipe"]) != _as_text(
            anomaly["contextual_anomalous_recipe"]
        ):
            _fail(f"Contextual pair {pair_id} must agree on its recipe mapping.")

    return recipe_context.loc[:, CONTEXTUAL_RECIPE_CONTEXT_COLUMNS].copy()


def _recipe_context_evidence_record(
    row: pd.Series,
    root_cause_id: str,
) -> dict[str, Any]:
    is_control = _as_text(row["is_contextual_control"]).strip().lower() in {
        "1",
        "true",
    }
    role = "normal control" if is_control else "contextual anomaly"
    return {
        "_sort_order": 1,
        "evidence_id": _as_text(row["recipe_context_evidence_id"]),
        "evidence_type": "recipe_context",
        "source_table": "synthetic_secom_v2",
        "source_record_id": _as_text(row["lot_id"]),
        "lot_id": _as_text(row["lot_id"]),
        "event_time": _as_text(row["event_time"]),
        "tool_id": _as_text(row["tool_id"]),
        "chamber_id": _as_text(row["chamber_id"]),
        "root_cause_id": root_cause_id,
        "is_shared_evidence": False,
        "evidence_summary": (
            f"Synthetic recipe-context {role}; "
            f"pair_id={_as_text(row['contextual_pair_id'])}; "
            f"recipe={_as_text(row['recipe_id'])}; "
            f"expected_status={_as_text(row['contextual_expected_status'])}; "
            f"sensor_columns={_as_text(row['contextual_sensor_columns'])}."
        ),
        "evidence_payload": _json_payload(
            row, CONTEXTUAL_RECIPE_CONTEXT_COLUMNS
        ),
    }


def _build_contextual_evidence_bundle(
    anomaly_lots: pd.DataFrame,
    tool_events: pd.DataFrame,
    recipe_context: pd.DataFrame,
    root_cause_id: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    context_spec = SCENARIO_SPECS["RCA_CONTEXTUAL_ANOMALY"]
    for event in tool_events.itertuples(index=False):
        records.append(
            _context_evidence_record(
                pd.Series(event._asdict()), root_cause_id, context_spec
            )
        )
    for context_row in recipe_context.itertuples(index=False):
        records.append(
            _recipe_context_evidence_record(
                pd.Series(context_row._asdict()), root_cause_id
            )
        )

    lot_payload_columns = [
        "lot_id",
        "event_time",
        "tool_id",
        "chamber_id",
        "recipe_id",
        "product_family",
        "anomaly_mechanism",
        "root_cause_id",
        "synthetic_evidence_id",
        "contextual_pair_id",
        "contextual_expected_status",
        "contextual_normal_recipe",
        "contextual_anomalous_recipe",
        "contextual_sensor_columns",
        "recipe_context_evidence_id",
    ]
    for lot in anomaly_lots.itertuples(index=False):
        row = pd.Series(lot._asdict())
        records.append(
            {
                "_sort_order": 2,
                "evidence_id": _as_text(row["synthetic_evidence_id"]),
                "evidence_type": "synthetic_lot",
                "source_table": "synthetic_secom_v2",
                "source_record_id": _as_text(row["lot_id"]),
                "lot_id": _as_text(row["lot_id"]),
                "event_time": _as_text(row["event_time"]),
                "tool_id": _as_text(row["tool_id"]),
                "chamber_id": _as_text(row["chamber_id"]),
                "root_cause_id": root_cause_id,
                "is_shared_evidence": False,
                "evidence_summary": "Synthetic contextual-anomaly lot evidence.",
                "evidence_payload": _json_payload(row, lot_payload_columns),
            }
        )

    bundle = pd.DataFrame(records)
    bundle = bundle.sort_values(
        ["_sort_order", "evidence_id"], kind="stable"
    ).reset_index(drop=True)
    return bundle.drop(columns="_sort_order")[EVIDENCE_BUNDLE_COLUMNS]


def _materialize_rca_contextual_anomaly(
    repo_root: Path,
    output_dir: Path | None,
) -> RcaEvaluationSummary:
    scenario_id = "RCA_CONTEXTUAL_ANOMALY"
    manifest_path = repo_root / "configs" / "synthetic_data_v2_stress_test_manifest.json"
    manifest = _load_json(manifest_path)
    scenario = _scenario_from_manifest(manifest, scenario_id)
    contract = _contextual_contract(scenario)

    lots_path = _resolve_repo_path(repo_root, contract["source_lots_path"])
    tool_events_path = _resolve_repo_path(
        repo_root, contract["source_tool_events_path"]
    )
    ground_truth_path = _resolve_repo_path(
        repo_root, contract["source_rca_ground_truth_path"]
    )
    for path in [lots_path, tool_events_path, ground_truth_path]:
        if not path.exists():
            _fail(f"Configured source file does not exist: {path}")

    lots = pd.read_csv(lots_path, keep_default_na=False)
    tool_events = pd.read_csv(tool_events_path, keep_default_na=False)
    ground_truth = pd.read_csv(ground_truth_path, keep_default_na=False)
    _require_columns(
        lots,
        [
            "lot_id",
            "event_time",
            "tool_id",
            "chamber_id",
            "anomaly_mechanism",
            "root_cause_id",
            "is_synthetic_anomaly",
            "synthetic_evidence_id",
        ],
        "Synthetic V2 lots",
    )
    _require_columns(
        ground_truth,
        [
            "case_id",
            "lot_id",
            "root_cause_id",
            "evidence_ids",
            "supports_abstention",
            "top3_acceptable_causes",
        ],
        "Synthetic V2 RCA ground truth",
    )
    _require_columns(tool_events, ["evidence_id"], "Synthetic V2 tool-event evidence")

    root_cause_id = _as_text(scenario["root_cause_id"])
    selected_truth = ground_truth.loc[
        ground_truth["root_cause_id"].astype(str).eq(root_cause_id)
    ].copy()
    selected_truth = _sort_table(selected_truth, ["case_id", "lot_id"])
    if len(selected_truth) != int(contract["expected_ground_truth_count"]):
        _fail(
            "Unexpected RCA ground-truth count: expected "
            f"{contract['expected_ground_truth_count']}, got {len(selected_truth)}."
        )
    if selected_truth["lot_id"].astype(str).duplicated().any():
        _fail(f"{scenario_id} ground truth must contain one case per lot.")
    if _true_mask(selected_truth["supports_abstention"]).any():
        _fail(f"{scenario_id} ground truth must not support abstention.")

    selected_lot_ids = set(selected_truth["lot_id"].astype(str))
    selected_lots = lots.loc[
        lots["lot_id"].astype(str).isin(selected_lot_ids)
    ].copy()
    selected_lots = _sort_table(selected_lots, ["event_time", "lot_id"])
    if len(selected_lots) != int(contract["expected_lot_count"]):
        _fail(
            "Unexpected RCA cohort lot count: expected "
            f"{contract['expected_lot_count']}, got {len(selected_lots)}."
        )
    if set(selected_lots["lot_id"].astype(str)) != selected_lot_ids:
        _fail("RCA ground-truth lot IDs do not match the selected cohort lots.")
    if set(selected_lots["root_cause_id"].astype(str)) != {root_cause_id}:
        _fail("Selected RCA lots have an unexpected root-cause ID.")
    if set(selected_lots["anomaly_mechanism"].astype(str)) != {
        "contextual_anomaly"
    }:
        _fail("Selected RCA lots must be contextual-anomaly lots only.")
    if not _true_mask(selected_lots["is_synthetic_anomaly"]).all():
        _fail("Selected RCA lots must all be synthetic anomalies.")

    recipe_context = _build_contextual_recipe_context(
        lots, selected_lots, contract
    )
    truth_evidence_by_lot = {
        _as_text(row.lot_id): _split_evidence_ids(row.evidence_ids)
        for row in selected_truth.itertuples(index=False)
    }
    truth_evidence_ids = set().union(*truth_evidence_by_lot.values())
    selected_tool_events = tool_events.loc[
        tool_events["evidence_id"].astype(str).isin(truth_evidence_ids)
    ].copy()
    selected_tool_events = _sort_table(
        selected_tool_events, ["start_time", "end_time", "evidence_id"]
    )
    required_tool_event_ids = _contract_evidence_ids(
        contract["required_tool_event_evidence_id"]
    )
    if len(selected_tool_events) != int(contract["expected_tool_event_count"]):
        _fail(
            "Unexpected RCA tool-event evidence count: expected "
            f"{contract['expected_tool_event_count']}, got {len(selected_tool_events)}."
        )
    if set(selected_tool_events["evidence_id"].astype(str)) != required_tool_event_ids:
        _fail("Contextual RCA tool-event evidence does not match the manifest contract.")

    recipe_evidence_ids = set(
        recipe_context["recipe_context_evidence_id"].map(_as_text)
    )
    if "" in recipe_evidence_ids or len(recipe_evidence_ids) != len(recipe_context):
        _fail("Contextual recipe-context evidence IDs must be present and unique.")
    expected_ground_truth_ids = required_tool_event_ids | recipe_evidence_ids
    if truth_evidence_ids != expected_ground_truth_ids:
        _fail(
            "Contextual RCA ground truth must cite the shared tool event and "
            "both recipe-context records for every pair."
        )

    context_by_pair = {
        _as_text(pair_id): group
        for pair_id, group in recipe_context.groupby("contextual_pair_id", sort=True)
    }
    for lot in selected_lots.itertuples(index=False):
        lot_id = _as_text(lot.lot_id)
        pair = context_by_pair[_as_text(lot.contextual_pair_id)]
        expected_case_ids = required_tool_event_ids | set(
            pair["recipe_context_evidence_id"].map(_as_text)
        )
        if truth_evidence_by_lot[lot_id] != expected_case_ids:
            _fail(
                f"Contextual RCA case for {lot_id} must cite its matched "
                "normal-control and anomaly recipe-context evidence."
            )

    lot_evidence_ids = set(selected_lots["synthetic_evidence_id"].map(_as_text))
    if "" in lot_evidence_ids or len(lot_evidence_ids) != len(selected_lots):
        _fail("Selected contextual RCA lots must each cite unique synthetic evidence.")
    anomaly_recipe_evidence_ids = set(
        recipe_context.loc[
            ~_true_mask(recipe_context["is_contextual_control"]),
            "recipe_context_evidence_id",
        ].map(_as_text)
    )
    if lot_evidence_ids != anomaly_recipe_evidence_ids:
        _fail(
            "Contextual synthetic-lot evidence must match the paired "
            "anomaly recipe-context evidence IDs."
        )
    evidence_ids = truth_evidence_ids | lot_evidence_ids

    evidence_bundle = _build_contextual_evidence_bundle(
        selected_lots,
        selected_tool_events,
        recipe_context,
        root_cause_id,
    )
    if set(evidence_bundle["evidence_id"].astype(str)) != evidence_ids:
        _fail("Written contextual RCA evidence bundle would not match validated IDs.")
    if evidence_bundle[["evidence_id", "evidence_type"]].duplicated().any():
        _fail("Contextual RCA evidence bundle contains duplicate typed evidence.")
    if len(evidence_bundle) != int(contract["expected_evidence_bundle_count"]):
        _fail(
            "Unexpected contextual RCA evidence-bundle row count: expected "
            f"{contract['expected_evidence_bundle_count']}, got {len(evidence_bundle)}."
        )

    if output_dir is None:
        output_dir = (
            repo_root / "data" / "synthetic" / "v2" / "scenarios" / scenario_id
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = SCENARIO_OUTPUT_FILES[scenario_id]
    _write_csv(selected_lots, output_dir / output_files["lots"])
    _write_csv(selected_truth, output_dir / output_files["ground_truth"])
    _write_csv(selected_tool_events, output_dir / output_files["tool_events"])
    _write_csv(recipe_context, output_dir / output_files["recipe_context"])
    _write_csv(evidence_bundle, output_dir / output_files["evidence_bundle"])

    generated_manifest = {
        "scenario": scenario,
        "cohort": {
            "lot_count": len(selected_lots),
            "ground_truth_count": len(selected_truth),
            "synthetic_anomaly_count": int(
                _true_mask(selected_lots["is_synthetic_anomaly"]).sum()
            ),
            "root_cause_id": root_cause_id,
            "expected_top_k": scenario["expected_top_k"],
            "expected_abstention": scenario["expected_abstention"],
        },
        "tool_event_evidence": {
            "source_table": "synthetic_tool_events",
            "row_count": len(selected_tool_events),
            "required_evidence_id": next(iter(required_tool_event_ids)),
        },
        "recipe_context_evidence": {
            "source_table": "synthetic_secom_v2",
            "row_count": len(recipe_context),
            "anomaly_record_count": len(selected_lots),
            "normal_control_record_count": int(
                _true_mask(recipe_context["is_contextual_control"]).sum()
            ),
            "pair_count": int(recipe_context["contextual_pair_id"].nunique()),
            "evidence_ids": sorted(recipe_evidence_ids),
        },
        "evidence_bundle": {
            "evidence_id_count": len(evidence_ids),
            "row_count": len(evidence_bundle),
            "synthetic_lot_count": len(selected_lots),
            "tool_event_count": len(selected_tool_events),
            "recipe_context_count": len(recipe_context),
            "required_evidence_types": scenario["required_evidence_types"],
        },
        "source_inputs": [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in [lots_path, tool_events_path, ground_truth_path]
        ],
        "disclaimer": contract["disclaimer"],
    }
    (output_dir / output_files["manifest"]).write_text(
        json.dumps(generated_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    return RcaEvaluationSummary(
        scenario_id=scenario_id,
        root_cause_id=root_cause_id,
        cohort_lot_count=len(selected_lots),
        ground_truth_count=len(selected_truth),
        tool_event_count=len(selected_tool_events),
        maintenance_count=0,
        context_evidence_type="tool_event",
        context_evidence_count=len(selected_tool_events),
        recipe_context_count=len(recipe_context),
        evidence_bundle_count=len(evidence_bundle),
        output_dir=output_dir,
    )


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n")


def materialize_rca_evaluation(
    repo_root: Path,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    output_dir: Path | None = None,
) -> RcaEvaluationSummary:
    """Materialize a supported RCA evaluation cohort deterministically."""
    if scenario_id not in SCENARIO_SPECS:
        _fail(
            "This checkpoint supports only "
            "RCA_ABRUPT_MEAN_SHIFT, RCA_GRADUAL_DEGRADATION, and "
            "RCA_VARIANCE_INSTABILITY, RCA_SENSOR_FAULT, and "
            "RCA_CONTEXTUAL_ANOMALY; "
            f"received {scenario_id}."
        )

    repo_root = repo_root.resolve()
    if scenario_id == "RCA_CONTEXTUAL_ANOMALY":
        return _materialize_rca_contextual_anomaly(repo_root, output_dir)

    manifest_path = repo_root / "configs" / "synthetic_data_v2_stress_test_manifest.json"
    manifest = _load_json(manifest_path)
    scenario = _scenario_from_manifest(manifest, scenario_id)
    contract, spec = _materialization_contract(scenario_id, scenario)

    lots_path = _resolve_repo_path(repo_root, contract["source_lots_path"])
    context_evidence_path = _resolve_repo_path(
        repo_root, contract[spec["source_path_key"]]
    )
    ground_truth_path = _resolve_repo_path(
        repo_root, contract["source_rca_ground_truth_path"]
    )
    for path in [lots_path, context_evidence_path, ground_truth_path]:
        if not path.exists():
            _fail(f"Configured source file does not exist: {path}")

    lots = pd.read_csv(lots_path, keep_default_na=False)
    context_evidence = pd.read_csv(context_evidence_path, keep_default_na=False)
    ground_truth = pd.read_csv(ground_truth_path, keep_default_na=False)
    selected_lots, selected_truth, selected_context_evidence, evidence_ids = (
        _validate_and_select_tables(
            lots,
            context_evidence,
            ground_truth,
            scenario,
            contract,
            spec,
        )
    )
    evidence_bundle = _build_evidence_bundle(
        selected_lots,
        selected_context_evidence,
        _as_text(scenario["root_cause_id"]),
        spec,
    )
    if set(evidence_bundle["evidence_id"].astype(str)) != evidence_ids:
        _fail("Written RCA evidence bundle would not match the validated IDs.")

    if output_dir is None:
        output_dir = (
            repo_root / "data" / "synthetic" / "v2" / "scenarios" / scenario_id
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = SCENARIO_OUTPUT_FILES[scenario_id]

    _write_csv(selected_lots, output_dir / output_files["lots"])
    _write_csv(selected_truth, output_dir / output_files["ground_truth"])
    _write_csv(
        selected_context_evidence,
        output_dir / output_files[spec["output_key"]],
    )
    _write_csv(evidence_bundle, output_dir / output_files["evidence_bundle"])

    context_type = spec["context_evidence_type"]
    required_context_evidence_ids = _contract_evidence_ids(
        contract.get(spec["required_evidence_id_key"])
    )
    generated_manifest = {
        "scenario": scenario,
        "cohort": {
            "lot_count": len(selected_lots),
            "ground_truth_count": len(selected_truth),
            "synthetic_anomaly_count": int(
                _true_mask(selected_lots["is_synthetic_anomaly"]).sum()
            ),
            "root_cause_id": _as_text(scenario["root_cause_id"]),
            "expected_top_k": scenario["expected_top_k"],
            "expected_abstention": scenario["expected_abstention"],
        },
        "context_evidence": {
            "evidence_type": context_type,
            "source_table": spec["source_table"],
            "row_count": len(selected_context_evidence),
            "required_evidence_id": (
                next(iter(required_context_evidence_ids))
                if len(required_context_evidence_ids) == 1
                else ""
            ),
            "required_evidence_ids": sorted(required_context_evidence_ids),
        },
        "evidence_bundle": {
            "evidence_id_count": len(evidence_ids),
            "synthetic_lot_count": len(selected_lots),
            "tool_event_count": (
                len(selected_context_evidence)
                if context_type == "tool_event"
                else 0
            ),
            "maintenance_count": (
                len(selected_context_evidence)
                if context_type == "maintenance"
                else 0
            ),
            "required_evidence_types": scenario["required_evidence_types"],
        },
        "source_inputs": [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in [lots_path, context_evidence_path, ground_truth_path]
        ],
        "disclaimer": contract["disclaimer"],
    }
    (output_dir / output_files["manifest"]).write_text(
        json.dumps(generated_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    return RcaEvaluationSummary(
        scenario_id=scenario_id,
        root_cause_id=_as_text(scenario["root_cause_id"]),
        cohort_lot_count=len(selected_lots),
        ground_truth_count=len(selected_truth),
        tool_event_count=(
            len(selected_context_evidence)
            if context_type == "tool_event"
            else 0
        ),
        maintenance_count=(
            len(selected_context_evidence)
            if context_type == "maintenance"
            else 0
        ),
        context_evidence_type=context_type,
        context_evidence_count=len(selected_context_evidence),
        recipe_context_count=0,
        evidence_bundle_count=len(evidence_bundle),
        output_dir=output_dir,
    )


def materialize_rca_abrupt_mean_shift(
    repo_root: Path,
    output_dir: Path | None = None,
) -> RcaEvaluationSummary:
    """Materialize the first manifest RCA evaluation scenario."""
    return materialize_rca_evaluation(
        repo_root=repo_root,
        scenario_id="RCA_ABRUPT_MEAN_SHIFT",
        output_dir=output_dir,
    )


def materialize_rca_gradual_degradation(
    repo_root: Path,
    output_dir: Path | None = None,
) -> RcaEvaluationSummary:
    """Materialize the gradual-degradation RCA evaluation cohort."""
    return materialize_rca_evaluation(
        repo_root=repo_root,
        scenario_id="RCA_GRADUAL_DEGRADATION",
        output_dir=output_dir,
    )


def materialize_rca_variance_instability(
    repo_root: Path,
    output_dir: Path | None = None,
) -> RcaEvaluationSummary:
    """Materialize the variance-instability RCA evaluation cohort."""
    return materialize_rca_evaluation(
        repo_root=repo_root,
        scenario_id="RCA_VARIANCE_INSTABILITY",
        output_dir=output_dir,
    )


def materialize_rca_sensor_fault(
    repo_root: Path,
    output_dir: Path | None = None,
) -> RcaEvaluationSummary:
    """Materialize the sensor-fault RCA evaluation cohort."""
    return materialize_rca_evaluation(
        repo_root=repo_root,
        scenario_id="RCA_SENSOR_FAULT",
        output_dir=output_dir,
    )


def materialize_rca_contextual_anomaly(
    repo_root: Path,
    output_dir: Path | None = None,
) -> RcaEvaluationSummary:
    """Materialize paired-control contextual-anomaly RCA evaluation data."""
    return materialize_rca_evaluation(
        repo_root=repo_root,
        scenario_id="RCA_CONTEXTUAL_ANOMALY",
        output_dir=output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a supported Synthetic Data V2 RCA evaluation cohort."
    )
    parser.add_argument(
        "--scenario-id",
        default=DEFAULT_SCENARIO_ID,
        choices=sorted(SCENARIO_SPECS),
        help="RCA scenario to materialize in this checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory used for deterministic test materialization.",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    try:
        summary = materialize_rca_evaluation(
            repo_root=repo_root,
            scenario_id=args.scenario_id,
            output_dir=args.output_dir,
        )
    except RcaEvaluationMaterializationError as error:
        print("SYNTHETIC_V2_RCA_EVALUATION_MATERIALIZATION_FAILED")
        print(f"- {error}")
        return 1

    print("SYNTHETIC_V2_RCA_EVALUATION_MATERIALIZATION_OK")
    print(f"Scenario: {summary.scenario_id}")
    print(f"Root cause: {summary.root_cause_id}")
    print(f"Cohort lots: {summary.cohort_lot_count}")
    print(f"RCA ground-truth cases: {summary.ground_truth_count}")
    print(
        f"{summary.context_evidence_type.replace('_', '-').title()} "
        f"evidence rows: {summary.context_evidence_count}"
    )
    if summary.recipe_context_count:
        print(f"Recipe-context evidence rows: {summary.recipe_context_count}")
    print(f"Evidence-bundle rows: {summary.evidence_bundle_count}")
    print(f"Output directory: {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
