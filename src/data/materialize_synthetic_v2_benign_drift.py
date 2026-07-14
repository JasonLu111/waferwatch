"""Materialize reproducible benign-drift Synthetic Data V2 cohorts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


MANIFEST_PATH = Path("configs/synthetic_data_v2_stress_test_manifest.json")
DEFAULT_SCENARIO_ID = "BENIGN_RECIPE_MIX"


SCENARIO_OUTPUT_FILES: dict[str, dict[str, str]] = {
    "BENIGN_RECIPE_MIX": {
        "lots": "benign_recipe_mix_lots.csv",
        "process_changes": "benign_recipe_mix_process_changes.csv",
        "rag_ground_truth": "benign_recipe_mix_rag_ground_truth.csv",
        "manifest": "benign_recipe_mix_cohort_manifest.json",
    },
    "BENIGN_PRODUCT_MIX": {
        "lots": "benign_product_mix_lots.csv",
        "process_changes": "benign_product_mix_process_changes.csv",
        "rag_ground_truth": "benign_product_mix_rag_ground_truth.csv",
        "manifest": "benign_product_mix_cohort_manifest.json",
    },
    "BENIGN_TOOL_REASSIGNMENT": {
        "lots": "benign_tool_reassignment_lots.csv",
        "process_changes": (
            "benign_tool_reassignment_process_changes.csv"
        ),
        "rag_ground_truth": (
            "benign_tool_reassignment_rag_ground_truth.csv"
        ),
        "manifest": "benign_tool_reassignment_cohort_manifest.json",
    },
}

# Kept for compatibility with the R5.19 recipe-mix test/import surface.
OUTPUT_FILES = SCENARIO_OUTPUT_FILES[DEFAULT_SCENARIO_ID]
BENIGN_RECIPE_MIX_OUTPUT_FILES = SCENARIO_OUTPUT_FILES["BENIGN_RECIPE_MIX"]
BENIGN_PRODUCT_MIX_OUTPUT_FILES = SCENARIO_OUTPUT_FILES["BENIGN_PRODUCT_MIX"]
BENIGN_TOOL_REASSIGNMENT_OUTPUT_FILES = SCENARIO_OUTPUT_FILES[
    "BENIGN_TOOL_REASSIGNMENT"
]
RAG_GROUND_TRUTH_COLUMNS = [
    "case_id",
    "lot_id",
    "scenario_id",
    "ground_truth_label",
    "expected_decision",
    "expected_abstention",
    "expected_root_cause_id",
    "suspected_cause",
    "recommended_action",
    "outcome",
    "evidence_ids",
    "evidence_type",
    "source_rca_case_id",
    "is_synthetic_anomaly",
    "is_benign_drift",
]


class BenignDriftMaterializationError(ValueError):
    """Raised when a benign-drift materialization contract is violated."""


# R5.19 compatibility alias.
BenignRecipeMixMaterializationError = BenignDriftMaterializationError


@dataclass(frozen=True)
class BenignDriftSummary:
    scenario_id: str
    benign_lot_count: int
    process_change_count: int
    rag_ground_truth_count: int
    output_dir: Path


# R5.19 compatibility alias.
BenignRecipeMixSummary = BenignDriftSummary


def _fail(message: str) -> None:
    raise BenignDriftMaterializationError(message)


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


def _true_mask(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_output_files(scenario_id: str) -> dict[str, str]:
    try:
        return SCENARIO_OUTPUT_FILES[scenario_id]
    except KeyError:
        _fail(f"Unsupported benign-drift scenario: {scenario_id}")


def _load_contract(
    repo_root: Path,
    scenario_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json((repo_root / MANIFEST_PATH).resolve())
    scenarios = manifest.get("scenarios")

    if not isinstance(scenarios, list):
        _fail("Stress-test manifest must contain a scenarios list.")

    matches = [
        item
        for item in scenarios
        if isinstance(item, dict) and item.get("id") == scenario_id
    ]

    if len(matches) != 1:
        _fail(
            f"Expected exactly one {scenario_id} scenario; found {len(matches)}."
        )

    scenario = matches[0]

    if scenario.get("scenario_type") != "benign_drift":
        _fail(f"{scenario_id} must use scenario_type=benign_drift.")

    benign_drift_type = scenario.get("benign_drift_type")

    if not isinstance(benign_drift_type, str) or not benign_drift_type:
        _fail(f"{scenario_id} requires a benign_drift_type.")

    if int(scenario.get("expected_is_synthetic_anomaly", -1)) != 0:
        _fail(f"{scenario_id} must require expected_is_synthetic_anomaly=0.")

    if scenario.get("expected_decision") != "abstain_or_no_escalation":
        _fail(f"{scenario_id} must require abstain_or_no_escalation.")

    contract = scenario.get("materialization")

    if not isinstance(contract, dict):
        _fail(f"{scenario_id} requires a materialization contract.")

    required_contract_keys = {
        "mode",
        "source_lots_path",
        "source_process_changes_path",
        "source_rca_ground_truth_path",
        "expected_lot_count",
        "process_change_id",
        "process_change_type",
        "process_evidence_id",
        "source_rca_case_id",
        "source_root_cause_id",
        "expected_source_rca_case_count",
        "expected_rag_ground_truth_count",
        "expected_abstention",
        "preserve_core_v2_outputs",
        "disclaimer",
    }
    missing_contract_keys = required_contract_keys.difference(contract)

    if missing_contract_keys:
        _fail(
            f"{scenario_id} materialization contract is missing: "
            f"{sorted(missing_contract_keys)}"
        )

    if contract["mode"] != "benign_drift_evaluation_cohort":
        _fail(
            f"{scenario_id} must use "
            "mode=benign_drift_evaluation_cohort."
        )

    if contract["process_change_type"] != benign_drift_type:
        _fail(
            f"{scenario_id} process_change_type must equal "
            "benign_drift_type."
        )

    if not bool(contract["expected_abstention"]):
        _fail(f"{scenario_id} must require RAG abstention.")

    if int(contract["expected_lot_count"]) <= 0:
        _fail("expected_lot_count must be positive.")

    if int(contract["expected_rag_ground_truth_count"]) != int(
        contract["expected_lot_count"]
    ):
        _fail(
            "expected_rag_ground_truth_count must equal "
            "expected_lot_count."
        )

    return scenario, contract


def _load_source_tables(
    repo_root: Path,
    contract: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    paths = {
        "lots": (
            repo_root / str(contract["source_lots_path"])
        ).resolve(),
        "process_changes": (
            repo_root / str(contract["source_process_changes_path"])
        ).resolve(),
        "rca_ground_truth": (
            repo_root / str(contract["source_rca_ground_truth_path"])
        ).resolve(),
    }
    source_tables: dict[str, pd.DataFrame] = {}

    for table_id, path in paths.items():
        if not path.is_file():
            _fail(f"Required source table does not exist: {path}")
        source_tables[table_id] = pd.read_csv(path, keep_default_na=False)

    return source_tables


def _select_source_records(
    source_tables: dict[str, pd.DataFrame],
    scenario: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    lots = source_tables["lots"]
    process_changes = source_tables["process_changes"]
    rca_ground_truth = source_tables["rca_ground_truth"]
    scenario_id = str(scenario["id"])
    benign_drift_type = str(scenario["benign_drift_type"])

    required_lot_columns = {
        "lot_id",
        "event_time",
        "is_synthetic_anomaly",
        "is_benign_drift",
        "benign_drift_type",
    }
    missing_lot_columns = required_lot_columns.difference(lots.columns)
    if missing_lot_columns:
        _fail(f"Source lots are missing columns: {sorted(missing_lot_columns)}")

    selected_lots = lots.loc[
        _true_mask(lots["is_benign_drift"])
        & lots["benign_drift_type"].astype(str).eq(benign_drift_type)
    ].copy()
    selected_lots = selected_lots.sort_values(
        ["event_time", "lot_id"], kind="mergesort"
    ).reset_index(drop=True)

    if len(selected_lots) != int(contract["expected_lot_count"]):
        _fail(
            f"Unexpected {scenario_id} lot count. Expected "
            f"{contract['expected_lot_count']}, got {len(selected_lots)}."
        )

    if _true_mask(selected_lots["is_synthetic_anomaly"]).any():
        _fail(f"{scenario_id} lots must not be synthetic anomalies.")

    if not _true_mask(selected_lots["is_benign_drift"]).all():
        _fail(f"{scenario_id} lots must all be benign drift.")

    required_change_columns = {
        "change_id",
        "change_type",
        "changed_at",
        "is_benign_drift",
        "evidence_id",
    }
    missing_change_columns = required_change_columns.difference(
        process_changes.columns
    )
    if missing_change_columns:
        _fail(
            "Source process changes are missing columns: "
            f"{sorted(missing_change_columns)}"
        )

    selected_changes = process_changes.loc[
        process_changes["change_id"].astype(str).eq(
            str(contract["process_change_id"])
        )
    ].copy()

    if len(selected_changes) != 1:
        _fail(
            f"{scenario_id} requires exactly one configured "
            "process-change evidence row."
        )

    selected_changes = selected_changes.sort_values(
        ["changed_at", "change_id"], kind="mergesort"
    ).reset_index(drop=True)
    change = selected_changes.iloc[0]

    if str(change["change_type"]) != str(contract["process_change_type"]):
        _fail("Configured process change has the wrong type.")

    if not _true_mask(
        pd.Series([change["is_benign_drift"]])
    ).iloc[0]:
        _fail("Configured process change must be marked benign drift.")

    if str(change["evidence_id"]) != str(contract["process_evidence_id"]):
        _fail("Configured process change has the wrong evidence ID.")

    required_rca_columns = {
        "case_id",
        "root_cause_id",
        "suspected_cause",
        "recommended_action",
        "outcome",
        "evidence_ids",
        "supports_abstention",
    }
    missing_rca_columns = required_rca_columns.difference(
        rca_ground_truth.columns
    )
    if missing_rca_columns:
        _fail(
            "Source RCA ground truth is missing columns: "
            f"{sorted(missing_rca_columns)}"
        )

    source_cases = rca_ground_truth.loc[
        rca_ground_truth["case_id"].astype(str).eq(
            str(contract["source_rca_case_id"])
        )
    ].copy()

    if len(source_cases) != int(contract["expected_source_rca_case_count"]):
        _fail(
            f"{scenario_id} has an unexpected source RCA case count."
        )

    source_case = source_cases.iloc[0]

    if str(source_case["root_cause_id"]) != str(
        contract["source_root_cause_id"]
    ):
        _fail("Configured source RCA case has the wrong root cause.")

    if not _true_mask(
        pd.Series([source_case["supports_abstention"]])
    ).iloc[0]:
        _fail("Configured source RCA case must support abstention.")

    if str(contract["process_evidence_id"]) not in str(
        source_case["evidence_ids"]
    ).split(";"):
        _fail("Configured source RCA case must cite process evidence.")

    return selected_lots, selected_changes, source_case


def _build_rag_ground_truth(
    lots: pd.DataFrame,
    source_case: pd.Series,
    scenario: dict[str, Any],
    contract: dict[str, Any],
) -> pd.DataFrame:
    evidence_id = str(contract["process_evidence_id"])
    scenario_id = str(scenario["id"])
    rows: list[dict[str, Any]] = []

    for lot in lots.itertuples(index=False):
        lot_id = str(lot.lot_id)
        rows.append(
            {
                "case_id": f"RAG_{scenario_id}_{lot_id}",
                "lot_id": lot_id,
                "scenario_id": scenario_id,
                "ground_truth_label": "benign_distribution_shift",
                "expected_decision": str(scenario["expected_decision"]),
                "expected_abstention": bool(
                    contract["expected_abstention"]
                ),
                "expected_root_cause_id": str(
                    contract["source_root_cause_id"]
                ),
                "suspected_cause": str(source_case["suspected_cause"]),
                "recommended_action": str(
                    source_case["recommended_action"]
                ),
                "outcome": str(source_case["outcome"]),
                "evidence_ids": evidence_id,
                "evidence_type": "process_change",
                "source_rca_case_id": str(source_case["case_id"]),
                "is_synthetic_anomaly": 0,
                "is_benign_drift": 1,
            }
        )

    return pd.DataFrame(rows, columns=RAG_GROUND_TRUTH_COLUMNS)


def _validate_derived_tables(
    lots: pd.DataFrame,
    process_changes: pd.DataFrame,
    rag_ground_truth: pd.DataFrame,
    source_case: pd.Series,
    scenario: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    if len(rag_ground_truth) != int(
        contract["expected_rag_ground_truth_count"]
    ):
        _fail("Unexpected benign RAG ground-truth count.")

    if list(rag_ground_truth.columns) != RAG_GROUND_TRUTH_COLUMNS:
        _fail("Benign RAG ground truth has an unexpected schema.")

    lot_ids = set(lots["lot_id"].astype(str))
    if set(rag_ground_truth["lot_id"].astype(str)) != lot_ids:
        _fail("Every benign lot must have one RAG ground-truth row.")

    if not rag_ground_truth["lot_id"].is_unique:
        _fail("Benign RAG ground-truth lot IDs must be unique.")

    if not rag_ground_truth["scenario_id"].eq(str(scenario["id"])).all():
        _fail("RAG ground truth has the wrong scenario ID.")

    if not rag_ground_truth["ground_truth_label"].eq(
        "benign_distribution_shift"
    ).all():
        _fail("RAG ground truth must label this as a benign distribution shift.")

    if not rag_ground_truth["expected_decision"].eq(
        str(scenario["expected_decision"])
    ).all():
        _fail("RAG ground truth has the wrong expected decision.")

    if not _true_mask(rag_ground_truth["expected_abstention"]).all():
        _fail("Every benign RAG case must require abstention.")

    if not rag_ground_truth["expected_root_cause_id"].eq(
        str(contract["source_root_cause_id"])
    ).all():
        _fail("RAG ground truth has the wrong root cause.")

    if not rag_ground_truth["evidence_ids"].eq(
        str(contract["process_evidence_id"])
    ).all():
        _fail("Every benign RAG case must cite the process-change evidence ID.")

    if not rag_ground_truth["evidence_type"].eq("process_change").all():
        _fail("Benign RAG evidence type must be process_change.")

    if not rag_ground_truth["source_rca_case_id"].eq(
        str(source_case["case_id"])
    ).all():
        _fail("RAG ground truth has the wrong source RCA case ID.")

    if _true_mask(rag_ground_truth["is_synthetic_anomaly"]).any():
        _fail("Benign RAG ground truth must not contain anomalies.")

    if not _true_mask(rag_ground_truth["is_benign_drift"]).all():
        _fail("Benign RAG ground truth must identify benign drift.")

    if len(process_changes) != 1:
        _fail(f"{scenario['id']} must contain one process change.")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_outputs(
    output_dir: Path,
    output_files: dict[str, str],
    lots: pd.DataFrame,
    process_changes: pd.DataFrame,
    rag_ground_truth: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "lots": lots,
        "process_changes": process_changes,
        "rag_ground_truth": rag_ground_truth,
    }
    output_hashes: dict[str, str] = {}

    for table_id, frame in frames.items():
        output_path = output_dir / output_files[table_id]
        _write_csv(frame, output_path)
        output_hashes[table_id] = _sha256(output_path)

    return output_hashes


def _write_cohort_manifest(
    repo_root: Path,
    output_dir: Path,
    output_files: dict[str, str],
    scenario: dict[str, Any],
    contract: dict[str, Any],
    summary: BenignDriftSummary,
    output_hashes: dict[str, str],
) -> None:
    source_paths = {
        "lots": repo_root / str(contract["source_lots_path"]),
        "process_changes": repo_root
        / str(contract["source_process_changes_path"]),
        "rca_ground_truth": repo_root
        / str(contract["source_rca_ground_truth_path"]),
    }
    generated_manifest = {
        "scenario_id": summary.scenario_id,
        "scenario_type": str(scenario["scenario_type"]),
        "description": str(scenario["description"]),
        "materialization_mode": str(contract["mode"]),
        "source_paths": {
            table_id: str(path.as_posix())
            for table_id, path in source_paths.items()
        },
        "source_sha256": {
            table_id: _sha256(path)
            for table_id, path in source_paths.items()
        },
        "cohort": {
            "benign_lot_count": summary.benign_lot_count,
            "synthetic_anomaly_lot_count": 0,
            "synthetic_anomaly_count": 0,
            "process_change_count": summary.process_change_count,
            "rag_ground_truth_count": summary.rag_ground_truth_count,
            "expected_is_synthetic_anomaly": int(
                scenario["expected_is_synthetic_anomaly"]
            ),
            "expected_decision": str(
                scenario["expected_decision"]
            ),
            "expected_abstention": bool(
                contract["expected_abstention"]
            ),
        },
        "benign_drift": {
            "type": str(scenario["benign_drift_type"]),
            "lot_count": summary.benign_lot_count,
            "is_synthetic_anomaly": 0,
        },
        "process_change_evidence": {
            "row_count": summary.process_change_count,
            "change_id": str(contract["process_change_id"]),
            "change_type": str(contract["process_change_type"]),
            "evidence_id": str(contract["process_evidence_id"]),
        },
        "process_change": {
            "row_count": summary.process_change_count,
            "change_id": str(contract["process_change_id"]),
            "change_type": str(contract["process_change_type"]),
            "evidence_id": str(contract["process_evidence_id"]),
        },
        "rag_ground_truth": {
            "row_count": summary.rag_ground_truth_count,
            "expected_decision": str(scenario["expected_decision"]),
            "expected_abstention": bool(contract["expected_abstention"]),
            "source_rca_case_id": str(contract["source_rca_case_id"]),
            "source_root_cause_id": str(contract["source_root_cause_id"]),
        },
        "output_sha256": output_hashes,
        "disclaimer": str(contract["disclaimer"]),
    }
    manifest_path = output_dir / output_files["manifest"]
    manifest_path.write_text(
        json.dumps(generated_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_benign_drift(
    repo_root: Path,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    output_dir: Path | None = None,
) -> BenignDriftSummary:
    """Materialize one configured benign-drift evaluation cohort."""
    repo_root = repo_root.resolve()
    output_files = _scenario_output_files(scenario_id)
    scenario, contract = _load_contract(repo_root, scenario_id)
    source_tables = _load_source_tables(repo_root, contract)
    lots, process_changes, source_case = _select_source_records(
        source_tables=source_tables,
        scenario=scenario,
        contract=contract,
    )
    rag_ground_truth = _build_rag_ground_truth(
        lots=lots,
        source_case=source_case,
        scenario=scenario,
        contract=contract,
    )
    _validate_derived_tables(
        lots=lots,
        process_changes=process_changes,
        rag_ground_truth=rag_ground_truth,
        source_case=source_case,
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
    summary = BenignDriftSummary(
        scenario_id=scenario_id,
        benign_lot_count=len(lots),
        process_change_count=len(process_changes),
        rag_ground_truth_count=len(rag_ground_truth),
        output_dir=output_dir,
    )
    output_hashes = _write_outputs(
        output_dir=output_dir,
        output_files=output_files,
        lots=lots,
        process_changes=process_changes,
        rag_ground_truth=rag_ground_truth,
    )
    _write_cohort_manifest(
        repo_root=repo_root,
        output_dir=output_dir,
        output_files=output_files,
        scenario=scenario,
        contract=contract,
        summary=summary,
        output_hashes=output_hashes,
    )
    return summary


def materialize_benign_recipe_mix(
    repo_root: Path,
    output_dir: Path | None = None,
) -> BenignDriftSummary:
    """Materialize the R5.19 recipe-mix cohort and RAG ground truth."""
    return materialize_benign_drift(
        repo_root=repo_root,
        scenario_id="BENIGN_RECIPE_MIX",
        output_dir=output_dir,
    )


def materialize_benign_product_mix(
    repo_root: Path,
    output_dir: Path | None = None,
) -> BenignDriftSummary:
    """Materialize the BENIGN_PRODUCT_MIX cohort and RAG ground truth."""
    return materialize_benign_drift(
        repo_root=repo_root,
        scenario_id="BENIGN_PRODUCT_MIX",
        output_dir=output_dir,
    )

def materialize_benign_tool_reassignment(
    repo_root: Path,
    output_dir: Path | None = None,
) -> BenignDriftSummary:
    """Materialize the BENIGN_TOOL_REASSIGNMENT cohort and RAG truth."""
    return materialize_benign_drift(
        repo_root=repo_root,
        scenario_id="BENIGN_TOOL_REASSIGNMENT",
        output_dir=output_dir,
    )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a benign-drift Synthetic Data V2 stress-test cohort."
    )
    parser.add_argument(
        "--scenario-id",
        choices=sorted(SCENARIO_OUTPUT_FILES),
        default=DEFAULT_SCENARIO_ID,
        help="Configured benign-drift scenario to materialize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the configured scenario directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = materialize_benign_drift(
            repo_root=Path.cwd(),
            scenario_id=args.scenario_id,
            output_dir=args.output_dir,
        )
    except (
        BenignDriftMaterializationError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(
            f"SYNTHETIC_V2_{args.scenario_id}_MATERIALIZATION_FAILED"
        )
        print(f"- {error}")
        return 1

    drift_label = str(args.scenario_id).replace("BENIGN_", "").lower().replace("_", "-")
    print(f"SYNTHETIC_V2_{args.scenario_id}_MATERIALIZATION_OK")
    print(f"Scenario: {summary.scenario_id}")
    print(f"Benign {drift_label} lots: {summary.benign_lot_count}")
    print("Synthetic anomaly lots: 0")
    print(f"Process-change evidence rows: {summary.process_change_count}")
    print(f"Benign RAG ground-truth rows: {summary.rag_ground_truth_count}")
    print("Expected decision: abstain_or_no_escalation")
    print(f"Output directory: {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
