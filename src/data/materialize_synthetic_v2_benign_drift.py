"""Materialize the BENIGN_RECIPE_MIX Synthetic Data V2 cohort."""

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
SCENARIO_ID = "BENIGN_RECIPE_MIX"
OUTPUT_FILES = {
    "lots": "benign_recipe_mix_lots.csv",
    "process_changes": "benign_recipe_mix_process_changes.csv",
    "rag_ground_truth": "benign_recipe_mix_rag_ground_truth.csv",
    "manifest": "benign_recipe_mix_cohort_manifest.json",
}
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


class BenignRecipeMixMaterializationError(ValueError):
    """Raised when the benign recipe-mix cohort is invalid."""


@dataclass(frozen=True)
class BenignRecipeMixSummary:
    """Stable summary for one benign recipe-mix cohort."""

    scenario_id: str
    benign_lot_count: int
    process_change_count: int
    rag_ground_truth_count: int
    output_dir: Path


def _fail(message: str) -> None:
    raise BenignRecipeMixMaterializationError(message)


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


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        _fail(f"Required CSV file does not exist: {path}")


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(repo_root / MANIFEST_PATH)
    scenarios = manifest.get("scenarios")

    if not isinstance(scenarios, list):
        _fail("Stress-test manifest must contain a scenarios list.")

    scenario = next(
        (
            item
            for item in scenarios
            if isinstance(item, dict)
            and item.get("id") == SCENARIO_ID
        ),
        None,
    )

    if scenario is None:
        _fail(f"Stress-test scenario is missing: {SCENARIO_ID}")

    if scenario.get("scenario_type") != "benign_drift":
        _fail(
            "BENIGN_RECIPE_MIX must use "
            "scenario_type=benign_drift."
        )

    if scenario.get("benign_drift_type") != "recipe_mix_change":
        _fail(
            "BENIGN_RECIPE_MIX must use "
            "benign_drift_type=recipe_mix_change."
        )

    if int(scenario.get("expected_is_synthetic_anomaly", -1)) != 0:
        _fail(
            "BENIGN_RECIPE_MIX must require "
            "expected_is_synthetic_anomaly=0."
        )

    if scenario.get("expected_decision") != (
        "abstain_or_no_escalation"
    ):
        _fail(
            "BENIGN_RECIPE_MIX must require "
            "abstain_or_no_escalation."
        )

    contract = scenario.get("materialization")

    if not isinstance(contract, dict):
        _fail(
            "BENIGN_RECIPE_MIX requires a materialization contract."
        )

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
    missing_contract_keys = required_contract_keys.difference(
        contract
    )

    if missing_contract_keys:
        _fail(
            "BENIGN_RECIPE_MIX materialization contract is "
            f"missing: {sorted(missing_contract_keys)}"
        )

    if contract["mode"] != "benign_drift_evaluation_cohort":
        _fail(
            "BENIGN_RECIPE_MIX must use "
            "mode=benign_drift_evaluation_cohort."
        )

    if contract["process_change_type"] != "recipe_mix_change":
        _fail(
            "BENIGN_RECIPE_MIX process_change_type must be "
            "recipe_mix_change."
        )

    if not bool(contract["expected_abstention"]):
        _fail(
            "BENIGN_RECIPE_MIX must require RAG abstention."
        )

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
            repo_root
            / str(contract["source_process_changes_path"])
        ).resolve(),
        "rca_ground_truth": (
            repo_root
            / str(contract["source_rca_ground_truth_path"])
        ).resolve(),
    }

    return {
        table_id: _read_csv(path)
        for table_id, path in paths.items()
    }


def _select_source_records(
    source_tables: dict[str, pd.DataFrame],
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    lots = source_tables["lots"]
    process_changes = source_tables["process_changes"]
    rca_ground_truth = source_tables["rca_ground_truth"]

    required_lot_columns = {
        "lot_id",
        "event_time",
        "is_synthetic_anomaly",
        "is_benign_drift",
        "benign_drift_type",
    }
    missing_lot_columns = required_lot_columns.difference(
        lots.columns
    )

    if missing_lot_columns:
        _fail(
            "Synthetic lots are missing columns: "
            f"{sorted(missing_lot_columns)}"
        )

    selected_lots = lots.loc[
        _true_mask(lots["is_benign_drift"])
        & lots["benign_drift_type"].astype(str).eq(
            "recipe_mix_change"
        )
    ].copy()

    selected_lots = selected_lots.sort_values(
        ["event_time", "lot_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(selected_lots) != int(
        contract["expected_lot_count"]
    ):
        _fail(
            "Unexpected BENIGN_RECIPE_MIX lot count. "
            f"Expected {contract['expected_lot_count']}, "
            f"got {len(selected_lots)}."
        )

    if _true_mask(selected_lots["is_synthetic_anomaly"]).any():
        _fail(
            "BENIGN_RECIPE_MIX lots must not be synthetic anomalies."
        )

    if not _true_mask(selected_lots["is_benign_drift"]).all():
        _fail(
            "BENIGN_RECIPE_MIX lots must all be benign drift."
        )

    selected_changes = process_changes.loc[
        process_changes["change_id"].astype(str).eq(
            str(contract["process_change_id"])
        )
    ].copy()

    if len(selected_changes) != 1:
        _fail(
            "BENIGN_RECIPE_MIX requires exactly one configured "
            "process-change evidence row."
        )

    selected_changes = selected_changes.sort_values(
        ["changed_at", "change_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    change = selected_changes.iloc[0]

    if str(change["change_type"]) != str(
        contract["process_change_type"]
    ):
        _fail("Configured process change has the wrong type.")

    if not _true_mask(
        pd.Series([change["is_benign_drift"]])
    ).iloc[0]:
        _fail(
            "Configured process change must be marked benign drift."
        )

    if str(change["evidence_id"]) != str(
        contract["process_evidence_id"]
    ):
        _fail("Configured process change has the wrong evidence ID.")

    selected_rca = rca_ground_truth.loc[
        rca_ground_truth["case_id"].astype(str).eq(
            str(contract["source_rca_case_id"])
        )
    ].copy()

    if len(selected_rca) != int(
        contract["expected_source_rca_case_count"]
    ):
        _fail(
            "Unexpected configured source RCA case count."
        )

    source_case = selected_rca.iloc[0]

    if str(source_case["root_cause_id"]) != str(
        contract["source_root_cause_id"]
    ):
        _fail("Configured source RCA has the wrong root cause.")

    if not _true_mask(
        pd.Series([source_case["supports_abstention"]])
    ).iloc[0]:
        _fail(
            "Configured source RCA must support abstention."
        )

    source_evidence_ids = {
        evidence_id.strip()
        for evidence_id in str(
            source_case["evidence_ids"]
        ).split(";")
        if evidence_id.strip()
    }

    if str(contract["process_evidence_id"]) not in source_evidence_ids:
        _fail(
            "Source RCA must cite the recipe-mix process evidence."
        )

    return selected_lots, selected_changes, source_case


def _build_rag_ground_truth(
    lots: pd.DataFrame,
    source_case: pd.Series,
    scenario: dict[str, Any],
    contract: dict[str, Any],
) -> pd.DataFrame:
    evidence_id = str(contract["process_evidence_id"])
    rows: list[dict[str, Any]] = []

    for lot in lots.itertuples(index=False):
        lot_id = str(lot.lot_id)

        rows.append(
            {
                "case_id": (
                    f"RAG_BENIGN_RECIPE_MIX_{lot_id}"
                ),
                "lot_id": lot_id,
                "scenario_id": str(scenario["id"]),
                "ground_truth_label": (
                    "benign_distribution_shift"
                ),
                "expected_decision": str(
                    scenario["expected_decision"]
                ),
                "expected_abstention": bool(
                    contract["expected_abstention"]
                ),
                "expected_root_cause_id": str(
                    contract["source_root_cause_id"]
                ),
                "suspected_cause": str(
                    source_case["suspected_cause"]
                ),
                "recommended_action": str(
                    source_case["recommended_action"]
                ),
                "outcome": str(source_case["outcome"]),
                "evidence_ids": evidence_id,
                "evidence_type": "process_change",
                "source_rca_case_id": str(
                    source_case["case_id"]
                ),
                "is_synthetic_anomaly": 0,
                "is_benign_drift": 1,
            }
        )

    rag_ground_truth = pd.DataFrame(
        rows,
        columns=RAG_GROUND_TRUTH_COLUMNS,
    )

    return rag_ground_truth


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
        _fail(
            "Benign RAG ground truth has an unexpected schema."
        )

    lot_ids = set(lots["lot_id"].astype(str))

    if set(rag_ground_truth["lot_id"].astype(str)) != lot_ids:
        _fail(
            "Every benign lot must have one RAG ground-truth row."
        )

    if not rag_ground_truth["lot_id"].is_unique:
        _fail("Benign RAG ground-truth lot IDs must be unique.")

    if not rag_ground_truth["scenario_id"].eq(
        str(scenario["id"])
    ).all():
        _fail("RAG ground truth has the wrong scenario ID.")

    if not rag_ground_truth["ground_truth_label"].eq(
        "benign_distribution_shift"
    ).all():
        _fail(
            "RAG ground truth must label this as a benign "
            "distribution shift."
        )

    if not rag_ground_truth["expected_decision"].eq(
        str(scenario["expected_decision"])
    ).all():
        _fail("RAG ground truth has the wrong expected decision.")

    if not _true_mask(
        rag_ground_truth["expected_abstention"]
    ).all():
        _fail(
            "Every benign RAG case must require abstention."
        )

    if not rag_ground_truth["expected_root_cause_id"].eq(
        str(contract["source_root_cause_id"])
    ).all():
        _fail("RAG ground truth has the wrong root cause.")

    if not rag_ground_truth["evidence_ids"].eq(
        str(contract["process_evidence_id"])
    ).all():
        _fail(
            "Every benign RAG case must cite the process-change "
            "evidence ID."
        )

    if not rag_ground_truth["evidence_type"].eq(
        "process_change"
    ).all():
        _fail("Benign RAG evidence type must be process_change.")

    if not rag_ground_truth["source_rca_case_id"].eq(
        str(source_case["case_id"])
    ).all():
        _fail("RAG ground truth has the wrong source RCA case ID.")

    if _true_mask(
        rag_ground_truth["is_synthetic_anomaly"]
    ).any():
        _fail(
            "Benign RAG ground truth must not contain anomalies."
        )

    if not _true_mask(
        rag_ground_truth["is_benign_drift"]
    ).all():
        _fail(
            "Benign RAG ground truth must identify benign drift."
        )

    if len(process_changes) != 1:
        _fail(
            "BENIGN_RECIPE_MIX must contain one process change."
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
    )


def _write_outputs(
    output_dir: Path,
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
        output_path = output_dir / OUTPUT_FILES[table_id]
        _write_csv(frame, output_path)
        output_hashes[table_id] = _sha256(output_path)

    return output_hashes


def _write_cohort_manifest(
    repo_root: Path,
    output_dir: Path,
    scenario: dict[str, Any],
    contract: dict[str, Any],
    summary: BenignRecipeMixSummary,
    output_hashes: dict[str, str],
) -> None:
    source_paths = {
        "lots": (
            repo_root / str(contract["source_lots_path"])
        ).resolve(),
        "process_changes": (
            repo_root
            / str(contract["source_process_changes_path"])
        ).resolve(),
        "rca_ground_truth": (
            repo_root
            / str(contract["source_rca_ground_truth_path"])
        ).resolve(),
    }

    generated_manifest = {
        "schema_version": "1.0",
        "scenario_id": summary.scenario_id,
        "scenario_type": scenario["scenario_type"],
        "source_sha256": {
            table_id: _sha256(path)
            for table_id, path in source_paths.items()
        },
        "cohort": {
            "benign_lot_count": summary.benign_lot_count,
            "synthetic_anomaly_count": 0,
            "benign_drift_type": scenario["benign_drift_type"],
            "expected_decision": scenario["expected_decision"],
        },
        "process_change_evidence": {
            "change_id": contract["process_change_id"],
            "change_type": contract["process_change_type"],
            "evidence_id": contract["process_evidence_id"],
        },
        "rag_ground_truth": {
            "row_count": summary.rag_ground_truth_count,
            "expected_abstention": bool(
                contract["expected_abstention"]
            ),
            "source_rca_case_id": contract["source_rca_case_id"],
            "source_root_cause_id": contract[
                "source_root_cause_id"
            ],
        },
        "output_sha256": output_hashes,
        "disclaimer": contract["disclaimer"],
    }

    manifest_path = output_dir / OUTPUT_FILES["manifest"]
    manifest_path.write_text(
        json.dumps(
            generated_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_benign_recipe_mix(
    repo_root: Path,
    output_dir: Path | None = None,
) -> BenignRecipeMixSummary:
    """Materialize the BENIGN_RECIPE_MIX cohort and RAG truth."""

    repo_root = repo_root.resolve()
    scenario, contract = _load_contract(repo_root)
    source_tables = _load_source_tables(
        repo_root=repo_root,
        contract=contract,
    )
    lots, process_changes, source_case = _select_source_records(
        source_tables=source_tables,
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
            / SCENARIO_ID
        )

    output_dir = output_dir.resolve()
    summary = BenignRecipeMixSummary(
        scenario_id=SCENARIO_ID,
        benign_lot_count=len(lots),
        process_change_count=len(process_changes),
        rag_ground_truth_count=len(rag_ground_truth),
        output_dir=output_dir,
    )

    output_hashes = _write_outputs(
        output_dir=output_dir,
        lots=lots,
        process_changes=process_changes,
        rag_ground_truth=rag_ground_truth,
    )
    _write_cohort_manifest(
        repo_root=repo_root,
        output_dir=output_dir,
        scenario=scenario,
        contract=contract,
        summary=summary,
        output_hashes=output_hashes,
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the BENIGN_RECIPE_MIX Synthetic Data V2 "
            "stress-test cohort."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "data/synthetic/v2/scenarios/BENIGN_RECIPE_MIX."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        summary = materialize_benign_recipe_mix(
            repo_root=Path.cwd(),
            output_dir=args.output_dir,
        )
    except (
        BenignRecipeMixMaterializationError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(
            "SYNTHETIC_V2_BENIGN_RECIPE_MIX_MATERIALIZATION_FAILED"
        )
        print(f"- {error}")
        return 1

    print(
        "SYNTHETIC_V2_BENIGN_RECIPE_MIX_MATERIALIZATION_OK"
    )
    print(f"Scenario: {summary.scenario_id}")
    print(f"Benign recipe-mix lots: {summary.benign_lot_count}")
    print("Synthetic anomaly lots: 0")
    print(
        "Process-change evidence rows: "
        f"{summary.process_change_count}"
    )
    print(
        "Benign RAG ground-truth rows: "
        f"{summary.rag_ground_truth_count}"
    )
    print("Expected decision: abstain_or_no_escalation")
    print(f"Output directory: {summary.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())