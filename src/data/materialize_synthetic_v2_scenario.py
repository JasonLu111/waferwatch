"""Materialize reproducible Synthetic Data V2 stress-test cohorts.

R5.11 deliberately supports only PREV_01. It does not train a model or
modify MLflow, FastAPI, Docker, or API schemas.
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

COHORT_OUTPUT_FILES = {
    "lots": "prev_01_lots.csv",
    "tool_events": "prev_01_tool_events.csv",
    "maintenance": "prev_01_maintenance.csv",
    "process_changes": "prev_01_process_changes.csv",
    "rca_ground_truth": "prev_01_rca_ground_truth.csv",
    "manifest": "prev_01_cohort_manifest.json",
}


class ScenarioMaterializationError(ValueError):
    """Raised when PREV_01 cannot be materialized safely."""


@dataclass(frozen=True)
class CohortSummary:
    scenario_id: str
    cohort_size: int
    synthetic_anomaly_count: int
    achieved_anomaly_rate: float
    mechanism_counts: dict[str, int]
    label_delay_counts: dict[str, int]
    output_dir: Path


def _fail(message: str) -> None:
    raise ScenarioMaterializationError(message)


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_base_tables(
    repo_root: Path,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    for table_id, relative_path in BASE_OUTPUT_FILES.items():
        path = repo_root / relative_path

        if not path.exists():
            _fail(f"Missing Synthetic Data V2 output: {path}")

        tables[table_id] = pd.read_csv(path)

    return tables


def _load_scenario(
    repo_root: Path,
    scenario_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = (
        repo_root
        / "configs"
        / "synthetic_data_v2_stress_test_manifest.json"
    )
    manifest = _load_json(manifest_path)

    scenario = next(
        (
            item
            for item in manifest["scenarios"]
            if item["id"] == scenario_id
        ),
        None,
    )

    if scenario is None:
        _fail(f"Unknown stress-test scenario: {scenario_id}")

    if scenario_id != "PREV_01":
        _fail(
            "R5.11 implements only PREV_01. Other stress-test "
            "scenarios remain manifest-only."
        )

    return manifest, scenario


def _materialization_contract(
    scenario: dict[str, Any],
) -> dict[str, Any]:
    required_keys = {
        "cohort_size",
        "anomalies_per_mechanism",
        "target_label_delay_counts",
        "exclude_benign_drift",
        "include_contextual_controls",
    }

    materialization = scenario.get("materialization", {})
    missing_keys = required_keys.difference(materialization)

    if missing_keys:
        _fail(
            "PREV_01 materialization contract is missing: "
            f"{sorted(missing_keys)}"
        )

    return materialization


def _scenario_seed(
    manifest: dict[str, Any],
    scenario_id: str,
) -> int:
    base_seed = int(manifest["research_scope"]["random_seed"])
    stable_offset = sum(
        (index + 1) * ord(character)
        for index, character in enumerate(scenario_id)
    )
    return base_seed + stable_offset


def _delay_counts(
    lots: pd.DataFrame,
    allowed_delays: list[int],
) -> dict[str, int]:
    delays = pd.to_numeric(
        lots["label_delay_hours"],
        errors="coerce",
    )

    return {
        str(delay): int(delays.eq(delay).sum())
        for delay in allowed_delays
    }


def _select_prev_01_lots(
    lots: pd.DataFrame,
    manifest: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], int]:
    required_columns = {
        "lot_id",
        "event_time",
        "label_delay_hours",
        "is_unseen_context",
        "is_synthetic_anomaly",
        "is_benign_drift",
        "anomaly_mechanism",
        "contextual_pair_id",
        "is_contextual_control",
    }
    missing_columns = required_columns.difference(lots.columns)

    if missing_columns:
        _fail(
            "Synthetic Data V2 lots table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    materialization = _materialization_contract(scenario)
    cohort_size = int(materialization["cohort_size"])
    anomalies_per_mechanism = int(
        materialization["anomalies_per_mechanism"]
    )
    allowed_delays = [
        int(delay)
        for delay in scenario["label_delay_hours"]
    ]
    target_delay_counts = {
        int(delay): int(count)
        for delay, count in materialization[
            "target_label_delay_counts"
        ].items()
    }
    included_mechanisms = [
        str(mechanism)
        for mechanism in scenario[
            "included_anomaly_mechanisms"
        ]
    ]

    if sum(target_delay_counts.values()) != cohort_size:
        _fail(
            "PREV_01 target_label_delay_counts must sum to cohort_size."
        )

    if set(target_delay_counts) != set(allowed_delays):
        _fail(
            "PREV_01 label-delay quotas must cover exactly 12h, 24h, "
            "and 48h."
        )

    expected_anomaly_count = (
        len(included_mechanisms) * anomalies_per_mechanism
    )

    target_rate = float(
        scenario["target_synthetic_anomaly_rate"]
    )

    if expected_anomaly_count / cohort_size != target_rate:
        _fail(
            "PREV_01 anomaly count and cohort size do not produce the "
            "configured anomaly prevalence."
        )

    seed = _scenario_seed(manifest, scenario["id"])
    unseen_mask = _true_mask(lots["is_unseen_context"])
    anomaly_mask = _true_mask(lots["is_synthetic_anomaly"])
    benign_mask = _true_mask(lots["is_benign_drift"])

    eligible_anomalies = lots.loc[
        anomaly_mask
        & ~unseen_mask
        & ~benign_mask
    ].copy()

    selected_anomaly_indices: list[int] = []

    for offset, mechanism_id in enumerate(included_mechanisms):
        mechanism_pool = eligible_anomalies.loc[
            eligible_anomalies["anomaly_mechanism"].eq(
                mechanism_id
            )
        ]

        if len(mechanism_pool) < anomalies_per_mechanism:
            _fail(
                f"PREV_01 needs {anomalies_per_mechanism} seen-context "
                f"{mechanism_id} lots, but only "
                f"{len(mechanism_pool)} are available."
            )

        selected_anomaly_indices.extend(
            mechanism_pool.sample(
                n=anomalies_per_mechanism,
                random_state=seed + offset,
            ).index.to_list()
        )

    selected_anomaly_indices = sorted(
        set(selected_anomaly_indices)
    )

    if len(selected_anomaly_indices) != expected_anomaly_count:
        _fail(
            "PREV_01 did not select the expected number of anomaly lots."
        )

    selected_anomalies = lots.loc[
        selected_anomaly_indices
    ].copy()

    normal_pool = lots.loc[
        ~anomaly_mask
        & ~unseen_mask
        & ~benign_mask
    ].copy()

    forced_control_indices: list[int] = []

    if bool(materialization["include_contextual_controls"]):
        selected_contextual_anomalies = selected_anomalies.loc[
            selected_anomalies["anomaly_mechanism"].eq(
                "contextual_anomaly"
            )
        ]
        contextual_pair_ids = set(
            selected_contextual_anomalies[
                "contextual_pair_id"
            ].astype(str)
        )

        if "" in contextual_pair_ids:
            _fail(
                "Selected contextual anomalies are missing pair IDs."
            )

        forced_controls = normal_pool.loc[
            _true_mask(normal_pool["is_contextual_control"])
            & normal_pool["contextual_pair_id"]
            .astype(str)
            .isin(contextual_pair_ids)
        ].copy()

        if len(forced_controls) != len(
            selected_contextual_anomalies
        ):
            _fail(
                "PREV_01 must include one Recipe A control for every "
                "selected contextual anomaly."
            )

        forced_control_indices = forced_controls.index.to_list()

    selected_normal_indices = set(forced_control_indices)
    selected_anomaly_delay_counts = _delay_counts(
        selected_anomalies,
        allowed_delays,
    )
    forced_control_delay_counts = _delay_counts(
        lots.loc[forced_control_indices],
        allowed_delays,
    )

    for offset, delay in enumerate(sorted(allowed_delays)):
        required_normal_count = (
            target_delay_counts[delay]
            - selected_anomaly_delay_counts[str(delay)]
            - forced_control_delay_counts[str(delay)]
        )

        if required_normal_count < 0:
            _fail(
                f"PREV_01 delay quota for {delay}h is overfilled by "
                "selected anomalies or mandatory contextual controls."
            )

        delay_pool = normal_pool.loc[
            pd.to_numeric(
                normal_pool["label_delay_hours"],
                errors="coerce",
            ).eq(delay)
            & ~normal_pool.index.isin(selected_normal_indices)
        ]

        if len(delay_pool) < required_normal_count:
            _fail(
                f"PREV_01 needs {required_normal_count} normal lots at "
                f"{delay}h, but only {len(delay_pool)} are available."
            )

        sampled_indices = delay_pool.sample(
            n=required_normal_count,
            random_state=seed + 100 + offset,
        ).index.to_list()
        selected_normal_indices.update(sampled_indices)

    if len(selected_normal_indices) != (
        cohort_size - expected_anomaly_count
    ):
        _fail(
            "PREV_01 did not select the required number of normal lots."
        )

    selected_indices = sorted(
        set(selected_anomaly_indices).union(selected_normal_indices)
    )
    cohort_lots = lots.loc[selected_indices].copy()

    if len(cohort_lots) != cohort_size:
        _fail(
            "PREV_01 cohort size does not match its materialization "
            "contract."
        )

    cohort_lots = cohort_lots.sort_values(
        ["event_time", "lot_id"]
    ).reset_index(drop=True)

    mechanism_counts = {
        mechanism_id: int(
            cohort_lots["anomaly_mechanism"]
            .eq(mechanism_id)
            .sum()
        )
        for mechanism_id in included_mechanisms
    }

    return (
        cohort_lots,
        mechanism_counts,
        _delay_counts(cohort_lots, allowed_delays),
        seed,
    )


def _filter_evidence_tables(
    source_tables: dict[str, pd.DataFrame],
    cohort_lots: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    anomaly_lot_ids = set(
        cohort_lots.loc[
            _true_mask(cohort_lots["is_synthetic_anomaly"]),
            "lot_id",
        ].astype(str)
    )

    cohort_rca = source_tables["rca_ground_truth"].loc[
        source_tables["rca_ground_truth"]["lot_id"]
        .astype(str)
        .isin(anomaly_lot_ids)
    ].copy()

    if len(cohort_rca) != len(anomaly_lot_ids):
        _fail(
            "Every selected anomaly must retain one RCA ground-truth case."
        )

    referenced_evidence_ids: set[str] = set()

    for evidence_ids in cohort_rca["evidence_ids"]:
        referenced_evidence_ids.update(
            _evidence_tokens(evidence_ids)
        )

    cohort_tool_events = source_tables["tool_events"].loc[
        source_tables["tool_events"]["evidence_id"]
        .astype(str)
        .isin(referenced_evidence_ids)
    ].copy()

    cohort_maintenance = source_tables["maintenance"].loc[
        source_tables["maintenance"]["evidence_id"]
        .astype(str)
        .isin(referenced_evidence_ids)
    ].copy()

    cohort_process_changes = source_tables["process_changes"].loc[
        source_tables["process_changes"]["evidence_id"]
        .astype(str)
        .isin(referenced_evidence_ids)
    ].copy()

    return {
        "lots": cohort_lots,
        "tool_events": cohort_tool_events,
        "maintenance": cohort_maintenance,
        "process_changes": cohort_process_changes,
        "rca_ground_truth": cohort_rca,
    }


def _write_cohort_tables(
    cohort_tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}

    for table_id, filename in COHORT_OUTPUT_FILES.items():
        if table_id == "manifest":
            continue

        path = output_dir / filename
        cohort_tables[table_id].to_csv(
            path,
            index=False,
            lineterminator="\n",
        )
        output_hashes[filename] = _sha256(path)

    return output_hashes


def _validate_cohort_evidence(
    cohort_tables: dict[str, pd.DataFrame],
) -> None:
    lots = cohort_tables["lots"]
    rca_ground_truth = cohort_tables["rca_ground_truth"]

    known_evidence_ids = set()

    for frame, column in (
        (cohort_tables["tool_events"], "evidence_id"),
        (cohort_tables["maintenance"], "evidence_id"),
        (cohort_tables["process_changes"], "evidence_id"),
        (lots, "synthetic_evidence_id"),
        (lots, "maintenance_evidence_id"),
        (lots, "recipe_context_evidence_id"),
    ):
        if column in frame.columns:
            known_evidence_ids.update(
                frame.loc[_non_empty(frame[column]), column]
                .astype(str)
                .str.strip()
            )

    for case_id, evidence_ids in zip(
        rca_ground_truth["case_id"].astype(str),
        rca_ground_truth["evidence_ids"],
    ):
        unknown_evidence_ids = _evidence_tokens(
            evidence_ids
        ).difference(known_evidence_ids)

        if unknown_evidence_ids:
            _fail(
                f"Cohort RCA case {case_id} references unavailable "
                f"evidence: {sorted(unknown_evidence_ids)}"
            )


def validate_prev_01_cohort(
    output_dir: Path,
    scenario: dict[str, Any],
) -> CohortSummary:
    """Validate a written PREV_01 cohort and its evidence closure."""
    output_dir = output_dir.resolve()
    materialization = _materialization_contract(scenario)
    required_paths = {
        table_id: output_dir / filename
        for table_id, filename in COHORT_OUTPUT_FILES.items()
    }

    missing_paths = [
        path
        for path in required_paths.values()
        if not path.exists()
    ]

    if missing_paths:
        _fail(
            "PREV_01 cohort outputs are missing: "
            f"{[str(path) for path in missing_paths]}"
        )

    cohort_tables = {
        table_id: pd.read_csv(path)
        for table_id, path in required_paths.items()
        if table_id != "manifest"
    }
    generated_manifest = _load_json(
        required_paths["manifest"]
    )
    lots = cohort_tables["lots"]

    cohort_size = int(materialization["cohort_size"])
    anomalies_per_mechanism = int(
        materialization["anomalies_per_mechanism"]
    )
    expected_delay_counts = {
        str(delay): int(count)
        for delay, count in materialization[
            "target_label_delay_counts"
        ].items()
    }
    included_mechanisms = [
        str(mechanism)
        for mechanism in scenario[
            "included_anomaly_mechanisms"
        ]
    ]

    if len(lots) != cohort_size:
        _fail(
            f"PREV_01 cohort size must be {cohort_size}, got {len(lots)}."
        )

    if _true_mask(lots["is_unseen_context"]).any():
        _fail("PREV_01 must contain seen-context lots only.")

    if _true_mask(lots["is_benign_drift"]).any():
        _fail("PREV_01 must exclude benign drift lots.")

    anomaly_mask = _true_mask(lots["is_synthetic_anomaly"])
    anomaly_lots = lots.loc[anomaly_mask].copy()
    expected_anomaly_count = (
        len(included_mechanisms) * anomalies_per_mechanism
    )

    if len(anomaly_lots) != expected_anomaly_count:
        _fail(
            "PREV_01 synthetic anomaly count does not match the "
            "materialization contract."
        )

    achieved_rate = len(anomaly_lots) / len(lots)

    if achieved_rate != float(
        scenario["target_synthetic_anomaly_rate"]
    ):
        _fail(
            "PREV_01 achieved anomaly rate does not match the manifest."
        )

    mechanism_counts = {
        mechanism_id: int(
            anomaly_lots["anomaly_mechanism"]
            .eq(mechanism_id)
            .sum()
        )
        for mechanism_id in included_mechanisms
    }

    if set(mechanism_counts.values()) != {
        anomalies_per_mechanism
    }:
        _fail(
            "PREV_01 must retain the configured anomaly count for every "
            "included mechanism."
        )

    observed_delay_counts = _delay_counts(
        lots,
        [12, 24, 48],
    )

    if observed_delay_counts != expected_delay_counts:
        _fail(
            "PREV_01 label-delay distribution does not match the "
            "manifest quotas."
        )

    contextual_anomalies = anomaly_lots.loc[
        anomaly_lots["anomaly_mechanism"].eq(
            "contextual_anomaly"
        )
    ]
    contextual_pair_ids = set(
        contextual_anomalies["contextual_pair_id"].astype(str)
    )
    contextual_controls = lots.loc[
        _true_mask(lots["is_contextual_control"])
        & lots["contextual_pair_id"]
        .astype(str)
        .isin(contextual_pair_ids)
    ]

    if len(contextual_controls) != len(contextual_anomalies):
        _fail(
            "PREV_01 must include one contextual Recipe A control per "
            "selected contextual anomaly."
        )

    anomaly_lot_ids = set(anomaly_lots["lot_id"].astype(str))
    rca_ground_truth = cohort_tables["rca_ground_truth"]

    if set(rca_ground_truth["lot_id"].astype(str)) != anomaly_lot_ids:
        _fail(
            "PREV_01 RCA ground truth must cover exactly its anomaly lots."
        )

    _validate_cohort_evidence(cohort_tables)

    if generated_manifest["scenario_id"] != "PREV_01":
        _fail("Generated cohort manifest must identify PREV_01.")

    if generated_manifest["cohort_size"] != cohort_size:
        _fail(
            "Generated cohort manifest has an incorrect cohort_size."
        )

    if generated_manifest["synthetic_anomaly_count"] != (
        expected_anomaly_count
    ):
        _fail(
            "Generated cohort manifest has an incorrect anomaly count."
        )

    return CohortSummary(
        scenario_id="PREV_01",
        cohort_size=cohort_size,
        synthetic_anomaly_count=expected_anomaly_count,
        achieved_anomaly_rate=achieved_rate,
        mechanism_counts=mechanism_counts,
        label_delay_counts=observed_delay_counts,
        output_dir=output_dir,
    )


def materialize_scenario(
    repo_root: Path,
    scenario_id: str = "PREV_01",
    output_dir: Path | None = None,
) -> CohortSummary:
    """Materialize, write, and validate the PREV_01 cohort."""
    repo_root = repo_root.resolve()
    manifest, scenario = _load_scenario(
        repo_root=repo_root,
        scenario_id=scenario_id,
    )
    source_tables = _load_base_tables(repo_root)

    (
        cohort_lots,
        mechanism_counts,
        label_delay_counts,
        scenario_seed,
    ) = _select_prev_01_lots(
        lots=source_tables["lots"],
        manifest=manifest,
        scenario=scenario,
    )

    cohort_tables = _filter_evidence_tables(
        source_tables=source_tables,
        cohort_lots=cohort_lots,
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
    output_hashes = _write_cohort_tables(
        cohort_tables=cohort_tables,
        output_dir=output_dir,
    )

    generated_manifest = {
        "schema_version": "1.0",
        "scenario_id": scenario_id,
        "source_stress_test_manifest": (
            "configs/synthetic_data_v2_stress_test_manifest.json"
        ),
        "source_lots": BASE_OUTPUT_FILES["lots"],
        "random_seed": scenario_seed,
        "cohort_size": len(cohort_lots),
        "synthetic_anomaly_count": int(
            _true_mask(cohort_lots["is_synthetic_anomaly"]).sum()
        ),
        "achieved_synthetic_anomaly_rate": (
            int(_true_mask(
                cohort_lots["is_synthetic_anomaly"]
            ).sum())
            / len(cohort_lots)
        ),
        "mechanism_counts": mechanism_counts,
        "label_delay_counts": label_delay_counts,
        "population_filters": [
            "is_unseen_context == 0",
            "is_benign_drift == 0",
        ],
        "output_sha256": output_hashes,
    }
    manifest_path = output_dir / COHORT_OUTPUT_FILES["manifest"]
    manifest_path.write_text(
        json.dumps(
            generated_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return validate_prev_01_cohort(
        output_dir=output_dir,
        scenario=scenario,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one reproducible WaferWatch Synthetic Data V2 "
            "stress-test cohort."
        )
    )
    parser.add_argument(
        "--scenario-id",
        default="PREV_01",
        help="Scenario ID. R5.11 supports PREV_01 only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "data/synthetic/v2/scenarios/PREV_01."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path.cwd().resolve()

    try:
        summary = materialize_scenario(
            repo_root=repo_root,
            scenario_id=args.scenario_id,
            output_dir=args.output_dir,
        )
    except (
        OSError,
        json.JSONDecodeError,
        ScenarioMaterializationError,
    ) as error:
        print("SYNTHETIC_V2_SCENARIO_MATERIALIZATION_FAILED")
        print(f"- {error}")
        return 1

    mechanism_text = ", ".join(
        f"{mechanism_id}={count}"
        for mechanism_id, count in summary.mechanism_counts.items()
    )
    delay_text = ", ".join(
        f"{delay}h={count}"
        for delay, count in summary.label_delay_counts.items()
    )

    print("SYNTHETIC_V2_SCENARIO_MATERIALIZATION_OK")
    print(f"Scenario: {summary.scenario_id}")
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
    print(f"Label-delay counts: {delay_text}")
    print("Population: seen context only; benign drift excluded.")
    print(
        "Contextual controls: included for selected contextual anomalies."
    )
    print(f"Output directory: {summary.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())