"""Validate the Synthetic Data V2 stress-test scenario manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_PREVALENCE_RATES = {0.01, 0.03, 0.05, 0.07}
EXPECTED_LABEL_DELAYS = {12, 24, 48}
EXPECTED_BENIGN_DRIFTS = {
    "recipe_mix_change",
    "product_mix_change",
    "tool_reassignment",
}
EXPECTED_ROOT_CAUSES = {
    "RC_PRESSURE_INSTABILITY": {
        "tool_event",
        "synthetic_lot",
    },
    "RC_COMPONENT_WEAR": {
        "maintenance",
        "synthetic_lot",
    },
    "RC_PROCESS_VARIABILITY": {
    "tool_event",
    "synthetic_lot",
    },
    "RC_SENSOR_HARDWARE": {
        "tool_event",
        "synthetic_lot",
    },
    "RC_RECIPE_CONTEXT_MISMATCH": {
        "tool_event",
        "recipe_context",
        "synthetic_lot",
    },
}
EXPECTED_ANOMALY_MECHANISMS = {
    "abrupt_mean_shift",
    "gradual_degradation",
    "variance_instability",
    "sensor_fault",
    "contextual_anomaly",
}
REQUIRED_OUTPUT_TABLES = {
    "synthetic_secom_v2.csv",
    "synthetic_tool_events.csv",
    "synthetic_maintenance.csv",
    "synthetic_process_changes.csv",
    "synthetic_rca_ground_truth.csv",
}


class StressTestManifestValidationError(ValueError):
    """Raised when the stress-test scenario contract is invalid."""


@dataclass(frozen=True)
class ManifestSummary:
    scenario_count: int
    prevalence_rates: list[float]
    label_delays: list[int]
    benign_drift_count: int
    root_cause_count: int


def _fail(message: str) -> None:
    raise StressTestManifestValidationError(message)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _require_keys(
    value: dict[str, Any],
    required_keys: set[str],
    label: str,
) -> None:
    missing_keys = required_keys.difference(value)

    if missing_keys:
        _fail(
            f"{label} is missing keys: {sorted(missing_keys)}"
        )


def _scenarios_by_type(
    scenarios: list[dict[str, Any]],
    scenario_type: str,
) -> list[dict[str, Any]]:
    return [
        scenario
        for scenario in scenarios
        if scenario.get("scenario_type") == scenario_type
    ]


def validate_manifest(manifest: dict[str, Any]) -> ManifestSummary:
    _require_keys(
        manifest,
        {
            "schema_version",
            "manifest_name",
            "status",
            "research_scope",
            "dataset_contract",
            "evaluation_contract",
            "scenarios",
        },
        "manifest",
    )

    if manifest["schema_version"] != "1.0":
        _fail("schema_version must be 1.0.")

    if manifest["status"] != "manifest_only":
        _fail(
            "status must be manifest_only; this checkpoint must not "
            "materialize cohorts or train models."
        )

    research_scope = manifest["research_scope"]
    _require_keys(
        research_scope,
        {
            "statement",
            "unit_of_analysis",
            "random_seed",
        },
        "research_scope",
    )

    if research_scope["unit_of_analysis"] != "lot":
        _fail("research_scope.unit_of_analysis must be lot.")

    dataset_contract = manifest["dataset_contract"]
    _require_keys(
        dataset_contract,
        {
            "generator_config",
            "quality_validator",
            "required_output_tables",
            "materialization_note",
        },
        "dataset_contract",
    )

    if dataset_contract["generator_config"] != (
        "configs/synthetic_data_v2.json"
    ):
        _fail(
            "dataset_contract.generator_config must reference the "
            "Synthetic Data V2 contract."
        )

    if set(dataset_contract["required_output_tables"]) != (
        REQUIRED_OUTPUT_TABLES
    ):
        _fail(
            "dataset_contract.required_output_tables does not match the "
            "five Synthetic Data V2 outputs."
        )

    evaluation_contract = manifest["evaluation_contract"]
    _require_keys(
        evaluation_contract,
        {
            "training_population_selector",
            "default_evaluation_selector",
            "false_alarm_budget_per_100_lots",
            "engineer_review_capacity_per_100_lots",
            "detection_metrics",
            "rca_metrics",
        },
        "evaluation_contract",
    )

    if evaluation_contract["training_population_selector"] != (
        "is_unseen_context == 0"
    ):
        _fail(
            "The training population must exclude unseen contexts."
        )

    if float(
        evaluation_contract["false_alarm_budget_per_100_lots"]
    ) <= 0:
        _fail("false_alarm_budget_per_100_lots must be positive.")

    if float(
        evaluation_contract["engineer_review_capacity_per_100_lots"]
    ) <= 0:
        _fail(
            "engineer_review_capacity_per_100_lots must be positive."
        )

    scenarios = manifest["scenarios"]

    if not isinstance(scenarios, list) or not scenarios:
        _fail("scenarios must be a non-empty list.")

    scenario_ids = [
        scenario.get("id")
        for scenario in scenarios
    ]

    if (
        any(not isinstance(scenario_id, str) or not scenario_id
            for scenario_id in scenario_ids)
        or len(scenario_ids) != len(set(scenario_ids))
    ):
        _fail("Every scenario must have one unique non-empty id.")

    prevalence_scenarios = _scenarios_by_type(
        scenarios,
        "anomaly_prevalence",
    )

    if len(prevalence_scenarios) != 4:
        _fail("Exactly four anomaly_prevalence scenarios are required.")

    observed_prevalence_rates = {
        float(scenario["target_synthetic_anomaly_rate"])
        for scenario in prevalence_scenarios
    }

    if observed_prevalence_rates != EXPECTED_PREVALENCE_RATES:
        _fail(
            "Anomaly prevalence scenarios must cover 1%, 3%, 5%, and 7%."
        )

    for scenario in prevalence_scenarios:
        _require_keys(
            scenario,
            {
                "id",
                "scenario_type",
                "evaluation_selector",
                "target_synthetic_anomaly_rate",
                "included_anomaly_mechanisms",
                "label_delay_hours",
                "sampling_method",
                "primary_metrics",
            },
            scenario["id"],
        )

        if set(scenario["included_anomaly_mechanisms"]) != (
            EXPECTED_ANOMALY_MECHANISMS
        ):
            _fail(
                f"{scenario['id']} must cover all five anomaly mechanisms."
            )

        if set(scenario["label_delay_hours"]) != EXPECTED_LABEL_DELAYS:
            _fail(
                f"{scenario['id']} must cover 12h, 24h, and 48h labels."
            )

        if scenario["sampling_method"] != (
            "stratified_resample_without_replacement"
        ):
            _fail(
                f"{scenario['id']} must use reproducible stratified sampling."
            )

    unseen_scenarios = _scenarios_by_type(
        scenarios,
        "unseen_context_generalization",
    )

    if len(unseen_scenarios) != 1:
        _fail(
            "Exactly one unseen_context_generalization scenario is required."
        )

    unseen_scenario = unseen_scenarios[0]
    _require_keys(
        unseen_scenario,
        {
            "id",
            "scenario_type",
            "training_selector",
            "evaluation_selector",
            "holdout_unit",
            "target_synthetic_anomaly_rate",
            "label_delay_hours",
            "primary_metrics",
        },
        unseen_scenario["id"],
    )

    if unseen_scenario["training_selector"] != (
        "is_unseen_context == 0"
    ) or unseen_scenario["evaluation_selector"] != (
        "is_unseen_context == 1"
    ):
        _fail(
            "Unseen-context scenario must use disjoint seen training and "
            "unseen evaluation selectors."
        )

    if set(unseen_scenario["holdout_unit"]) != {
        "tool_id",
        "chamber_id",
    }:
        _fail(
            "Unseen-context scenario must hold out tool_id and chamber_id."
        )

    label_delay_scenarios = _scenarios_by_type(
        scenarios,
        "label_delay",
    )

    if len(label_delay_scenarios) != 3:
        _fail("Exactly three label_delay scenarios are required.")

    observed_delay_conditions = {
        tuple(scenario["label_delay_hours"])
        for scenario in label_delay_scenarios
    }

    if observed_delay_conditions != {
        (12,),
        (24,),
        (48,),
    }:
        _fail(
            "Label-delay scenarios must separately cover 12h, 24h, and 48h."
        )

    benign_scenarios = _scenarios_by_type(
        scenarios,
        "benign_drift",
    )

    if len(benign_scenarios) != 3:
        _fail("Exactly three benign_drift scenarios are required.")

    observed_benign_drifts = {
        scenario.get("benign_drift_type")
        for scenario in benign_scenarios
    }

    if observed_benign_drifts != EXPECTED_BENIGN_DRIFTS:
        _fail(
            "Benign drift scenarios must cover recipe mix, product mix, "
            "and tool reassignment."
        )

    for scenario in benign_scenarios:
        _require_keys(
            scenario,
            {
                "id",
                "scenario_type",
                "evaluation_selector",
                "benign_drift_type",
                "expected_is_synthetic_anomaly",
                "expected_decision",
                "primary_metrics",
            },
            scenario["id"],
        )

        if scenario["expected_is_synthetic_anomaly"] != 0:
            _fail(
                f"{scenario['id']} must not label benign drift as anomaly."
            )

        if scenario["expected_decision"] != (
            "abstain_or_no_escalation"
        ):
            _fail(
                f"{scenario['id']} must evaluate abstention/no escalation."
            )

    root_cause_scenarios = _scenarios_by_type(
        scenarios,
        "root_cause",
    )

    if len(root_cause_scenarios) != 5:
        _fail("Exactly five root_cause scenarios are required.")

    observed_root_causes = {
        scenario.get("root_cause_id")
        for scenario in root_cause_scenarios
    }

    if observed_root_causes != set(EXPECTED_ROOT_CAUSES):
        _fail(
            "Root-cause scenarios must cover all five synthetic "
            "anomaly root causes."
        )

    for scenario in root_cause_scenarios:
        _require_keys(
            scenario,
            {
                "id",
                "scenario_type",
                "root_cause_id",
                "required_evidence_types",
                "expected_top_k",
                "expected_abstention",
            },
            scenario["id"],
        )

        root_cause_id = scenario["root_cause_id"]

        if set(scenario["required_evidence_types"]) != (
            EXPECTED_ROOT_CAUSES[root_cause_id]
        ):
            _fail(
                f"{scenario['id']} has an invalid evidence contract."
            )

        if scenario["expected_top_k"] != [1, 3]:
            _fail(
                f"{scenario['id']} must evaluate both Top-1 and Top-3 RCA."
            )

        if scenario["expected_abstention"] is not False:
            _fail(
                f"{scenario['id']} must not abstain on a known anomaly."
            )

    return ManifestSummary(
        scenario_count=len(scenarios),
        prevalence_rates=sorted(observed_prevalence_rates),
        label_delays=sorted(EXPECTED_LABEL_DELAYS),
        benign_drift_count=len(benign_scenarios),
        root_cause_count=len(root_cause_scenarios),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the WaferWatch Synthetic Data V2 stress-test "
            "scenario manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "configs/synthetic_data_v2_stress_test_manifest.json"
        ),
        help="Manifest path relative to the repository root.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()

    try:
        manifest = load_manifest(manifest_path)
        summary = validate_manifest(manifest)
    except (
        OSError,
        json.JSONDecodeError,
        StressTestManifestValidationError,
    ) as error:
        print("SYNTHETIC_V2_STRESS_TEST_MANIFEST_FAILED")
        print(f"- {error}")
        return 1

    prevalence_text = ", ".join(
        f"{int(rate * 100)}%"
        for rate in summary.prevalence_rates
    )
    delay_text = ", ".join(
        f"{delay}h"
        for delay in summary.label_delays
    )

    print("SYNTHETIC_V2_STRESS_TEST_MANIFEST_OK")
    print(f"Manifest: {manifest_path}")
    print(f"Scenarios: {summary.scenario_count}")
    print(f"Anomaly prevalence: {prevalence_text}")
    print("Unseen tool/chamber scenario: enabled")
    print(f"Label-delay conditions: {delay_text}")
    print(f"Benign drift scenarios: {summary.benign_drift_count}")
    print(f"RCA scenarios: {summary.root_cause_count}")
    print("Status: manifest only; no model or API changes were made.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())