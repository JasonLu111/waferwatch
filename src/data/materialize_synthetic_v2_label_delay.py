"""Materialize fixed Synthetic Data V2 label-delay cohorts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


MANIFEST_PATH = Path(
    "configs/synthetic_data_v2_stress_test_manifest.json"
)
DEFAULT_SCENARIO_ID = "LABEL_DELAY_12H"
SCENARIO_ID = DEFAULT_SCENARIO_ID
SUPPORTED_SCENARIO_DELAYS = {
    "LABEL_DELAY_12H": 12,
    "LABEL_DELAY_24H": 24,
}
TABLE_IDS = (
    "lots",
    "tool_events",
    "maintenance",
    "process_changes",
    "rca_ground_truth",
)


def cohort_output_files(scenario_id: str) -> dict[str, str]:
    """Return stable output filenames for one label-delay scenario."""

    prefix = scenario_id.lower()

    return {
        "lots": f"{prefix}_lots.csv",
        "tool_events": f"{prefix}_tool_events.csv",
        "maintenance": f"{prefix}_maintenance.csv",
        "process_changes": f"{prefix}_process_changes.csv",
        "rca_ground_truth": f"{prefix}_rca_ground_truth.csv",
        "manifest": f"{prefix}_cohort_manifest.json",
    }


# Backward-compatible constant used by the existing 12-hour test.
OUTPUT_FILES = cohort_output_files(DEFAULT_SCENARIO_ID)


class LabelDelayMaterializationError(ValueError):
    """Raised when a fixed label-delay cohort is invalid."""


@dataclass(frozen=True)
class LabelDelaySummary:
    """Stable summary for one fixed label-delay cohort."""

    scenario_id: str
    source_cohort_id: str
    cohort_size: int
    synthetic_anomaly_count: int
    achieved_anomaly_rate: float
    mechanism_counts: dict[str, int]
    label_delay_hours: int
    output_dir: Path


def _fail(message: str) -> None:
    raise LabelDelayMaterializationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail(f"Required JSON file does not exist: {path}")
    except json.JSONDecodeError as error:
        _fail(f"Invalid JSON in {path}: {error}")

    if not isinstance(payload, dict):
        _fail(f"JSON root must be an object: {path}")

    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _true_mask(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()

    return normalized.isin({"1", "1.0", "true", "yes"})


def _load_contract(
    repo_root: Path,
    scenario_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scenario_id not in SUPPORTED_SCENARIO_DELAYS:
        _fail(
            "Supported label-delay scenarios are: "
            f"{sorted(SUPPORTED_SCENARIO_DELAYS)}."
        )

    manifest = _load_json(repo_root / MANIFEST_PATH)
    scenarios = manifest.get("scenarios")

    if not isinstance(scenarios, list):
        _fail("Stress-test manifest must contain a scenarios list.")

    scenario = next(
        (
            item
            for item in scenarios
            if isinstance(item, dict)
            and item.get("id") == scenario_id
        ),
        None,
    )

    if scenario is None:
        _fail(f"Stress-test scenario is missing: {scenario_id}")

    if scenario.get("scenario_type") != "label_delay":
        _fail(f"{scenario_id} must use scenario_type=label_delay.")

    if scenario.get("evaluation_selector") != (
        "is_unseen_context == 0"
    ):
        _fail(f"{scenario_id} must evaluate seen contexts only.")

    raw_delays = scenario.get("label_delay_hours")

    if not isinstance(raw_delays, list):
        _fail(f"{scenario_id} label_delay_hours must be a list.")

    expected_delay = SUPPORTED_SCENARIO_DELAYS[scenario_id]
    delays = [int(delay) for delay in raw_delays]

    if delays != [expected_delay]:
        _fail(
            f"{scenario_id} must define exactly one "
            f"{expected_delay}-hour delay condition."
        )

    contract = scenario.get("materialization")

    if not isinstance(contract, dict):
        _fail(f"{scenario_id} requires a materialization contract.")

    required_contract_keys = {
        "mode",
        "source_cohort_id",
        "source_cohort_path",
        "cohort_size",
        "target_synthetic_anomaly_count",
        "target_anomaly_mechanism_counts",
        "fixed_label_delay_hours",
        "label_available_at_column",
        "exclude_benign_drift",
        "preserve_source_lot_ids",
        "preserve_source_sensor_values",
        "preserve_source_context",
        "preserve_core_v2_outputs",
        "quality_label_policy",
        "disclaimer",
    }
    missing_contract_keys = required_contract_keys.difference(
        contract
    )

    if missing_contract_keys:
        _fail(
            f"{scenario_id} materialization contract is missing: "
            f"{sorted(missing_contract_keys)}"
        )

    if contract["mode"] != "fixed_label_delay_overlay":
        _fail(
            f"{scenario_id} must use "
            "mode=fixed_label_delay_overlay."
        )

    if str(contract["source_cohort_id"]) != "PREV_03":
        _fail(
            f"{scenario_id} must use PREV_03 as its fixed "
            "peer cohort."
        )

    if int(contract["fixed_label_delay_hours"]) != expected_delay:
        _fail(
            f"{scenario_id} fixed_label_delay_hours must be "
            f"{expected_delay}."
        )

    mechanism_counts = contract[
        "target_anomaly_mechanism_counts"
    ]

    if not isinstance(mechanism_counts, dict):
        _fail(
            "target_anomaly_mechanism_counts must be an object."
        )

    target_anomaly_count = int(
        contract["target_synthetic_anomaly_count"]
    )
    configured_mechanism_total = sum(
        int(count)
        for count in mechanism_counts.values()
    )

    if configured_mechanism_total != target_anomaly_count:
        _fail(
            "target_anomaly_mechanism_counts must sum to "
            "target_synthetic_anomaly_count."
        )

    return scenario, contract


def _load_source_tables(
    repo_root: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], Path]:
    source_id = str(contract["source_cohort_id"])
    source_dir = (
        repo_root / str(contract["source_cohort_path"])
    ).resolve()
    source_files = cohort_output_files(source_id)
    required_paths = {
        table_id: source_dir / filename
        for table_id, filename in source_files.items()
    }
    missing_paths = [
        str(path)
        for path in required_paths.values()
        if not path.exists()
    ]

    if missing_paths:
        _fail(
            "Label-delay materialization requires the committed "
            f"{source_id} cohort. Missing: {missing_paths}"
        )

    source_manifest = _load_json(required_paths["manifest"])

    if source_manifest.get("scenario_id") != source_id:
        _fail(
            f"Source cohort manifest does not describe {source_id}."
        )

    source_tables = {
        table_id: pd.read_csv(required_paths[table_id])
        for table_id in TABLE_IDS
    }

    return source_tables, source_dir


def _mechanism_counts(lots: pd.DataFrame) -> dict[str, int]:
    anomaly_mask = _true_mask(lots["is_synthetic_anomaly"])
    counts = (
        lots.loc[anomaly_mask, "anomaly_mechanism"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    return {
        str(mechanism): int(count)
        for mechanism, count in counts.items()
    }


def _assert_lot_contract(
    lots: pd.DataFrame,
    rca_ground_truth: pd.DataFrame,
    scenario: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    required_columns = {
        "lot_id",
        "event_time",
        "label_delay_hours",
        "is_unseen_context",
        "is_synthetic_anomaly",
        "is_benign_drift",
        "anomaly_mechanism",
    }
    missing_columns = required_columns.difference(lots.columns)

    if missing_columns:
        _fail(
            "Label-delay lots are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(lots) != int(contract["cohort_size"]):
        _fail("Unexpected label-delay cohort size.")

    if _true_mask(lots["is_unseen_context"]).any():
        _fail("Label-delay cohorts must contain seen contexts only.")

    if bool(contract["exclude_benign_drift"]) and _true_mask(
        lots["is_benign_drift"]
    ).any():
        _fail("Label-delay cohorts must exclude benign drift lots.")

    anomaly_mask = _true_mask(lots["is_synthetic_anomaly"])
    anomaly_lots = lots.loc[anomaly_mask].copy()
    expected_anomaly_count = int(
        contract["target_synthetic_anomaly_count"]
    )

    if len(anomaly_lots) != expected_anomaly_count:
        _fail("Unexpected synthetic anomaly count.")

    achieved_anomaly_rate = len(anomaly_lots) / len(lots)
    expected_anomaly_rate = float(
        scenario["target_synthetic_anomaly_rate"]
    )

    if abs(achieved_anomaly_rate - expected_anomaly_rate) > 1e-12:
        _fail("Unexpected synthetic anomaly prevalence.")

    expected_mechanism_counts = {
        str(mechanism): int(count)
        for mechanism, count in contract[
            "target_anomaly_mechanism_counts"
        ].items()
    }
    actual_mechanism_counts = _mechanism_counts(lots)

    if actual_mechanism_counts != expected_mechanism_counts:
        _fail(
            "Unexpected mechanism counts. "
            f"Expected {expected_mechanism_counts}, "
            f"got {actual_mechanism_counts}."
        )

    if "lot_id" not in rca_ground_truth.columns:
        _fail("RCA ground truth must contain lot_id.")

    anomaly_lot_ids = {
        str(lot_id)
        for lot_id in anomaly_lots["lot_id"]
    }
    rca_lot_ids = {
        str(lot_id)
        for lot_id in rca_ground_truth["lot_id"].dropna()
        if str(lot_id).strip()
    }

    if len(rca_ground_truth) != len(anomaly_lots):
        _fail(
            "RCA ground truth must contain one case for every "
            "synthetic anomaly lot."
        )

    if rca_lot_ids != anomaly_lot_ids:
        _fail(
            "RCA ground truth must cover exactly the synthetic "
            "anomaly lots."
        )


def _apply_fixed_label_delay(
    source_lots: pd.DataFrame,
    scenario_id: str,
    contract: dict[str, Any],
) -> pd.DataFrame:
    label_available_at_column = str(
        contract["label_available_at_column"]
    )
    new_columns = {
        "source_label_delay_hours",
        "label_delay_scenario_id",
        label_available_at_column,
    }
    collisions = new_columns.intersection(source_lots.columns)

    if collisions:
        _fail(
            "Source PREV_03 lots already contain reserved "
            "label-delay columns: "
            f"{sorted(collisions)}"
        )

    source_delays = pd.to_numeric(
        source_lots["label_delay_hours"],
        errors="coerce",
    )

    if source_delays.isna().any():
        _fail(
            "Source PREV_03 has invalid label_delay_hours values."
        )

    event_times = pd.to_datetime(
        source_lots["event_time"],
        errors="coerce",
    )

    if event_times.isna().any():
        _fail("Source PREV_03 has invalid event_time values.")

    fixed_delay_hours = int(
        contract["fixed_label_delay_hours"]
    )
    materialized_lots = source_lots.copy()

    materialized_lots["source_label_delay_hours"] = (
        source_delays.astype(int)
    )
    materialized_lots["label_delay_hours"] = fixed_delay_hours
    materialized_lots[label_available_at_column] = (
        event_times
        + pd.to_timedelta(fixed_delay_hours, unit="h")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    materialized_lots["label_delay_scenario_id"] = scenario_id

    return materialized_lots


def _validate_materialized_tables(
    source_tables: dict[str, pd.DataFrame],
    cohort_tables: dict[str, pd.DataFrame],
    scenario: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    source_lots = source_tables["lots"]
    materialized_lots = cohort_tables["lots"]
    label_available_at_column = str(
        contract["label_available_at_column"]
    )
    fixed_delay_hours = int(
        contract["fixed_label_delay_hours"]
    )

    _assert_lot_contract(
        lots=materialized_lots,
        rca_ground_truth=cohort_tables["rca_ground_truth"],
        scenario=scenario,
        contract=contract,
    )

    preserved_source_columns = [
        column
        for column in source_lots.columns
        if column != "label_delay_hours"
    ]

    try:
        pd.testing.assert_frame_equal(
            materialized_lots[
                preserved_source_columns
            ].reset_index(drop=True),
            source_lots[
                preserved_source_columns
            ].reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        _fail(
            "Label-delay overlay changed source values other than "
            f"label_delay_hours: {error}"
        )

    actual_delays = pd.to_numeric(
        materialized_lots["label_delay_hours"],
        errors="coerce",
    )

    if actual_delays.isna().any() or not actual_delays.eq(
        fixed_delay_hours
    ).all():
        _fail(
            "Every lot must use the configured fixed label delay."
        )

    expected_source_delays = pd.to_numeric(
        source_lots["label_delay_hours"],
        errors="coerce",
    ).astype(int)
    actual_source_delays = pd.to_numeric(
        materialized_lots["source_label_delay_hours"],
        errors="coerce",
    )

    if actual_source_delays.isna().any() or not actual_source_delays.eq(
        expected_source_delays
    ).all():
        _fail(
            "source_label_delay_hours must preserve PREV_03 "
            "label-delay values."
        )

    expected_available_at = (
        pd.to_datetime(
            materialized_lots["event_time"],
            errors="coerce",
        )
        + pd.to_timedelta(fixed_delay_hours, unit="h")
    )
    actual_available_at = pd.to_datetime(
        materialized_lots[label_available_at_column],
        errors="coerce",
    )

    if actual_available_at.isna().any() or not actual_available_at.eq(
        expected_available_at
    ).all():
        _fail(
            "quality_label_available_at must equal event_time "
            "plus the configured delay."
        )

    if not materialized_lots[
        "label_delay_scenario_id"
    ].eq(str(scenario["id"])).all():
        _fail(
            "Every lot must identify its label-delay scenario."
        )

    for table_id in TABLE_IDS[1:]:
        try:
            pd.testing.assert_frame_equal(
                cohort_tables[table_id].reset_index(drop=True),
                source_tables[table_id].reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError as error:
            _fail(
                f"Label-delay materialization must preserve "
                f"{table_id}: {error}"
            )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
    )


def _write_cohort_tables(
    cohort_tables: dict[str, pd.DataFrame],
    output_dir: Path,
    output_files: dict[str, str],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}

    for table_id in TABLE_IDS:
        output_path = output_dir / output_files[table_id]
        _write_csv(cohort_tables[table_id], output_path)
        output_hashes[table_id] = _sha256(output_path)

    return output_hashes


def _write_generated_manifest(
    output_dir: Path,
    output_files: dict[str, str],
    source_dir: Path,
    scenario: dict[str, Any],
    contract: dict[str, Any],
    summary: LabelDelaySummary,
    output_hashes: dict[str, str],
) -> None:
    source_id = str(contract["source_cohort_id"])
    source_files = cohort_output_files(source_id)
    source_lots_path = source_dir / source_files["lots"]
    source_manifest_path = source_dir / source_files["manifest"]

    generated_manifest = {
        "schema_version": "1.0",
        "scenario_id": summary.scenario_id,
        "scenario_type": scenario["scenario_type"],
        "source_cohort": {
            "id": source_id,
            "path": str(contract["source_cohort_path"]),
            "lots_sha256": _sha256(source_lots_path),
            "manifest_sha256": _sha256(source_manifest_path),
        },
        "materialization": {
            "mode": contract["mode"],
            "fixed_label_delay_hours": (
                summary.label_delay_hours
            ),
            "label_available_at_column": contract[
                "label_available_at_column"
            ],
            "quality_label_policy": contract[
                "quality_label_policy"
            ],
            "preserve_source_lot_ids": bool(
                contract["preserve_source_lot_ids"]
            ),
            "preserve_source_sensor_values": bool(
                contract["preserve_source_sensor_values"]
            ),
            "preserve_source_context": bool(
                contract["preserve_source_context"]
            ),
            "preserve_core_v2_outputs": bool(
                contract["preserve_core_v2_outputs"]
            ),
        },
        "cohort": {
            "size": summary.cohort_size,
            "synthetic_anomaly_count": (
                summary.synthetic_anomaly_count
            ),
            "synthetic_anomaly_rate": (
                summary.achieved_anomaly_rate
            ),
            "mechanism_counts": summary.mechanism_counts,
            "benign_drift_included": False,
            "unseen_context_included": False,
        },
        "output_sha256": output_hashes,
        "disclaimer": contract["disclaimer"],
    }

    manifest_path = output_dir / output_files["manifest"]
    manifest_path.write_text(
        json.dumps(
            generated_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_label_delay(
    repo_root: Path,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    output_dir: Path | None = None,
) -> LabelDelaySummary:
    """Materialize and validate one supported fixed-delay cohort."""

    repo_root = repo_root.resolve()
    scenario, contract = _load_contract(
        repo_root=repo_root,
        scenario_id=scenario_id,
    )
    source_tables, source_dir = _load_source_tables(
        repo_root=repo_root,
        contract=contract,
    )

    _assert_lot_contract(
        lots=source_tables["lots"],
        rca_ground_truth=source_tables["rca_ground_truth"],
        scenario=scenario,
        contract=contract,
    )

    cohort_tables = {
        table_id: table.copy()
        for table_id, table in source_tables.items()
    }
    cohort_tables["lots"] = _apply_fixed_label_delay(
        source_lots=source_tables["lots"],
        scenario_id=scenario_id,
        contract=contract,
    )

    _validate_materialized_tables(
        source_tables=source_tables,
        cohort_tables=cohort_tables,
        scenario=scenario,
        contract=contract,
    )

    if output_dir is None:
        output_dir = (
            repo_root
            / "data"
            / "synthetic"
            / "v2"
            / "scenarios"
            / scenario_id
        )

    output_dir = output_dir.resolve()
    output_files = cohort_output_files(scenario_id)
    lots = cohort_tables["lots"]
    anomaly_count = int(
        _true_mask(lots["is_synthetic_anomaly"]).sum()
    )
    summary = LabelDelaySummary(
        scenario_id=scenario_id,
        source_cohort_id=str(contract["source_cohort_id"]),
        cohort_size=len(lots),
        synthetic_anomaly_count=anomaly_count,
        achieved_anomaly_rate=anomaly_count / len(lots),
        mechanism_counts=_mechanism_counts(lots),
        label_delay_hours=int(
            contract["fixed_label_delay_hours"]
        ),
        output_dir=output_dir,
    )

    output_hashes = _write_cohort_tables(
        cohort_tables=cohort_tables,
        output_dir=output_dir,
        output_files=output_files,
    )
    _write_generated_manifest(
        output_dir=output_dir,
        output_files=output_files,
        source_dir=source_dir,
        scenario=scenario,
        contract=contract,
        summary=summary,
        output_hashes=output_hashes,
    )

    return summary


def materialize_label_delay_12h(
    repo_root: Path,
    output_dir: Path | None = None,
) -> LabelDelaySummary:
    """Backward-compatible 12-hour materializer."""

    return materialize_label_delay(
        repo_root=repo_root,
        scenario_id="LABEL_DELAY_12H",
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a fixed Synthetic Data V2 "
            "label-delay stress-test cohort."
        )
    )
    parser.add_argument(
        "--scenario-id",
        choices=sorted(SUPPORTED_SCENARIO_DELAYS),
        default=DEFAULT_SCENARIO_ID,
        help="Supported label-delay scenario ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "data/synthetic/v2/scenarios/<scenario-id>."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        summary = materialize_label_delay(
            repo_root=Path.cwd(),
            scenario_id=args.scenario_id,
            output_dir=args.output_dir,
        )
    except (
        LabelDelayMaterializationError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print("SYNTHETIC_V2_LABEL_DELAY_MATERIALIZATION_FAILED")
        print(f"- {error}")
        return 1

    mechanism_text = ", ".join(
        f"{mechanism}={count}"
        for mechanism, count in summary.mechanism_counts.items()
    )

    print("SYNTHETIC_V2_LABEL_DELAY_MATERIALIZATION_OK")
    print(f"Scenario: {summary.scenario_id}")
    print(f"Source cohort: {summary.source_cohort_id}")
    print(f"Cohort size: {summary.cohort_size}")
    print(
        "Synthetic anomaly prevalence: "
        f"{summary.achieved_anomaly_rate:.2%}"
    )
    print(
        "Synthetic anomaly lots: "
        f"{summary.synthetic_anomaly_count}"
    )
    print(f"Mechanism counts: {mechanism_text}")
    print(
        "Fixed quality-label delay: "
        f"{summary.label_delay_hours}h"
    )
    print(
        "Population: same seen-context PREV_03 lots; "
        "benign drift excluded."
    )
    print(f"Output directory: {summary.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())