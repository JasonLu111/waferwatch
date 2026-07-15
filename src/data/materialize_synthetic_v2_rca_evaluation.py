"""Materialize the first Synthetic Data V2 RCA evaluation cohort.

This checkpoint deliberately supports only ``RCA_ABRUPT_MEAN_SHIFT``.  The
cohort is a deterministic, evidence-complete subset of the core V2 outputs:
47 abrupt-shift lots, their 47 RCA ground-truth cases, and the one shared tool
alarm plus lot-level synthetic evidence cited by those cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SCENARIO_ID = "RCA_ABRUPT_MEAN_SHIFT"

SCENARIO_OUTPUT_FILES: dict[str, dict[str, str]] = {
    DEFAULT_SCENARIO_ID: {
        "lots": "rca_abrupt_mean_shift_lots.csv",
        "ground_truth": "rca_abrupt_mean_shift_ground_truth.csv",
        "tool_events": "rca_abrupt_mean_shift_tool_events.csv",
        "evidence_bundle": "rca_abrupt_mean_shift_evidence_bundle.csv",
        "manifest": "rca_abrupt_mean_shift_cohort_manifest.json",
    }
}
RCA_ABRUPT_MEAN_SHIFT_OUTPUT_FILES = SCENARIO_OUTPUT_FILES[
    DEFAULT_SCENARIO_ID
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
    """Raised when the RCA cohort contract cannot be materialized safely."""


@dataclass(frozen=True)
class RcaEvaluationSummary:
    scenario_id: str
    root_cause_id: str
    cohort_lot_count: int
    ground_truth_count: int
    tool_event_count: int
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


def _materialization_contract(scenario: dict[str, Any]) -> dict[str, Any]:
    expected_root_cause_id = "RC_PRESSURE_INSTABILITY"
    if scenario.get("scenario_type") != "root_cause":
        _fail("RCA_ABRUPT_MEAN_SHIFT must have scenario_type root_cause.")
    if scenario.get("root_cause_id") != expected_root_cause_id:
        _fail(
            "RCA_ABRUPT_MEAN_SHIFT must target "
            f"{expected_root_cause_id}."
        )
    if set(scenario.get("required_evidence_types", [])) != {
        "tool_event",
        "synthetic_lot",
    }:
        _fail(
            "RCA_ABRUPT_MEAN_SHIFT must require tool_event and "
            "synthetic_lot evidence."
        )
    if tuple(scenario.get("expected_top_k", [])) != (1, 3):
        _fail("RCA_ABRUPT_MEAN_SHIFT must declare expected_top_k [1, 3].")
    if scenario.get("expected_abstention") is not False:
        _fail("RCA_ABRUPT_MEAN_SHIFT must not allow abstention.")

    contract = scenario.get("materialization")
    if not isinstance(contract, dict):
        _fail(
            "RCA_ABRUPT_MEAN_SHIFT requires a materialization object in "
            "the stress-test manifest."
        )

    required_fields = [
        "mode",
        "source_lots_path",
        "source_tool_events_path",
        "source_rca_ground_truth_path",
        "expected_lot_count",
        "expected_ground_truth_count",
        "expected_tool_event_count",
        "expected_evidence_bundle_count",
        "required_tool_event_evidence_id",
        "preserve_core_v2_outputs",
        "disclaimer",
    ]
    missing = [field for field in required_fields if field not in contract]
    if missing:
        _fail(
            "RCA_ABRUPT_MEAN_SHIFT materialization is missing fields: "
            f"{', '.join(missing)}."
        )
    if contract["mode"] != "rca_evaluation_cohort":
        _fail("RCA materialization mode must be rca_evaluation_cohort.")
    if contract["preserve_core_v2_outputs"] is not True:
        _fail("RCA materialization must preserve core V2 outputs.")
    if not _as_text(contract["disclaimer"]):
        _fail("RCA materialization must include a research-only disclaimer.")
    return contract


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
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_and_select_tables(
    lots: pd.DataFrame,
    tool_events: pd.DataFrame,
    ground_truth: pd.DataFrame,
    scenario: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    root_cause_id = _as_text(scenario["root_cause_id"])
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
    _require_columns(tool_events, ["evidence_id"], "Synthetic V2 tool events")

    selected_truth = ground_truth.loc[
        ground_truth["root_cause_id"].astype(str).eq(root_cause_id)
    ].copy()
    selected_truth = _sort_table(selected_truth, ["case_id", "lot_id"])

    expected_truth_count = int(contract["expected_ground_truth_count"])
    if len(selected_truth) != expected_truth_count:
        _fail(
            "Unexpected RCA ground-truth count: "
            f"expected {expected_truth_count}, got {len(selected_truth)}."
        )
    if selected_truth["lot_id"].astype(str).duplicated().any():
        _fail("RCA ground truth must contain one case per abrupt-shift lot.")
    if _true_mask(selected_truth["supports_abstention"]).any():
        _fail("RCA_ABRUPT_MEAN_SHIFT ground truth must not support abstention.")

    selected_lot_ids = set(selected_truth["lot_id"].astype(str))
    selected_lots = lots.loc[
        lots["lot_id"].astype(str).isin(selected_lot_ids)
    ].copy()
    selected_lots = _sort_table(selected_lots, ["event_time", "lot_id"])

    expected_lot_count = int(contract["expected_lot_count"])
    if len(selected_lots) != expected_lot_count:
        _fail(
            "Unexpected RCA cohort lot count: "
            f"expected {expected_lot_count}, got {len(selected_lots)}."
        )
    if set(selected_lots["lot_id"].astype(str)) != selected_lot_ids:
        _fail("RCA ground-truth lot IDs do not match the selected cohort lots.")
    if set(selected_lots["root_cause_id"].astype(str)) != {root_cause_id}:
        _fail("Selected RCA lots have an unexpected root-cause ID.")
    if set(selected_lots["anomaly_mechanism"].astype(str)) != {
        "abrupt_mean_shift"
    }:
        _fail("Selected RCA lots must be abrupt_mean_shift anomalies only.")
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

    selected_tool_events = tool_events.loc[
        tool_events["evidence_id"].astype(str).isin(truth_evidence_ids)
    ].copy()
    selected_tool_events = _sort_table(
        selected_tool_events,
        ["event_time", "event_id", "evidence_id"],
    )

    expected_tool_event_count = int(contract["expected_tool_event_count"])
    if len(selected_tool_events) != expected_tool_event_count:
        _fail(
            "Unexpected RCA tool-event evidence count: "
            f"expected {expected_tool_event_count}, got {len(selected_tool_events)}."
        )
    required_event_evidence_id = _as_text(
        contract["required_tool_event_evidence_id"]
    )
    if set(selected_tool_events["evidence_id"].astype(str)) != {
        required_event_evidence_id
    }:
        _fail("RCA tool-event evidence does not match the manifest contract.")

    bundle_evidence_ids = lot_evidence_ids | set(
        selected_tool_events["evidence_id"].astype(str)
    )
    if bundle_evidence_ids != truth_evidence_ids:
        _fail(
            "RCA evidence bundle would not cover every ground-truth evidence ID."
        )
    expected_bundle_count = int(contract["expected_evidence_bundle_count"])
    if len(bundle_evidence_ids) != expected_bundle_count:
        _fail(
            "Unexpected RCA evidence-bundle count: "
            f"expected {expected_bundle_count}, got {len(bundle_evidence_ids)}."
        )
    return selected_lots, selected_truth, selected_tool_events, bundle_evidence_ids


def _build_evidence_bundle(
    lots: pd.DataFrame,
    tool_events: pd.DataFrame,
    root_cause_id: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for event in tool_events.itertuples(index=False):
        row = pd.Series(event._asdict())
        event_type = _first_present(row, ["event_type", "event_name", "event_category"])
        description = _first_present(row, ["description", "event_description", "message"])
        summary = "; ".join(value for value in [event_type, description] if value)
        records.append(
            {
                "_sort_order": 0,
                "evidence_id": _as_text(row["evidence_id"]),
                "evidence_type": "tool_event",
                "source_table": "synthetic_tool_events",
                "source_record_id": _first_present(
                    row, ["event_id", "tool_event_id", "evidence_id"]
                ),
                "lot_id": _first_present(row, ["related_lot_id", "lot_id"]),
                "event_time": _first_present(row, ["event_time", "occurred_at"]),
                "tool_id": _first_present(row, ["tool_id"]),
                "chamber_id": _first_present(row, ["chamber_id"]),
                "root_cause_id": root_cause_id,
                "is_shared_evidence": True,
                "evidence_summary": summary or "Synthetic tool-event evidence.",
                "evidence_payload": _json_payload(row, list(row.index)),
            }
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
        "abrupt_shift_sigma",
        "synthetic_evidence_id",
    ]
    for lot in lots.itertuples(index=False):
        row = pd.Series(lot._asdict())
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
                "evidence_summary": (
                    "Synthetic abrupt mean-shift lot evidence; "
                    f"sensors={_first_present(row, ['injected_sensor_columns'])}; "
                    f"sigma={_first_present(row, ['abrupt_shift_sigma'])}."
                ),
                "evidence_payload": _json_payload(row, lot_payload_columns),
            }
        )

    bundle = pd.DataFrame(records)
    bundle = bundle.sort_values(
        ["_sort_order", "evidence_id"], kind="stable"
    ).reset_index(drop=True)
    return bundle.drop(columns="_sort_order")[EVIDENCE_BUNDLE_COLUMNS]


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n")


def materialize_rca_evaluation(
    repo_root: Path,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    output_dir: Path | None = None,
) -> RcaEvaluationSummary:
    """Materialize the supported root-cause evaluation cohort deterministically."""
    if scenario_id != DEFAULT_SCENARIO_ID:
        _fail(
            "This checkpoint supports only RCA_ABRUPT_MEAN_SHIFT; "
            f"received {scenario_id}."
        )

    repo_root = repo_root.resolve()
    manifest_path = repo_root / "configs" / "synthetic_data_v2_stress_test_manifest.json"
    manifest = _load_json(manifest_path)
    scenario = _scenario_from_manifest(manifest, scenario_id)
    contract = _materialization_contract(scenario)

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
    selected_lots, selected_truth, selected_tool_events, evidence_ids = (
        _validate_and_select_tables(
            lots,
            tool_events,
            ground_truth,
            scenario,
            contract,
        )
    )
    evidence_bundle = _build_evidence_bundle(
        selected_lots,
        selected_tool_events,
        _as_text(scenario["root_cause_id"]),
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
    _write_csv(selected_tool_events, output_dir / output_files["tool_events"])
    _write_csv(evidence_bundle, output_dir / output_files["evidence_bundle"])

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
        "evidence_bundle": {
            "evidence_id_count": len(evidence_ids),
            "tool_event_count": len(selected_tool_events),
            "synthetic_lot_count": len(selected_lots),
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
        root_cause_id=_as_text(scenario["root_cause_id"]),
        cohort_lot_count=len(selected_lots),
        ground_truth_count=len(selected_truth),
        tool_event_count=len(selected_tool_events),
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
        scenario_id=DEFAULT_SCENARIO_ID,
        output_dir=output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a Synthetic Data V2 RCA evaluation cohort."
    )
    parser.add_argument(
        "--scenario-id",
        default=DEFAULT_SCENARIO_ID,
        help="Only RCA_ABRUPT_MEAN_SHIFT is implemented in this checkpoint.",
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
    print(f"Tool-event evidence rows: {summary.tool_event_count}")
    print(f"Evidence-bundle rows: {summary.evidence_bundle_count}")
    print(f"Output directory: {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())