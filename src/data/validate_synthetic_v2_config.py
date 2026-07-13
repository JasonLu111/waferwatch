from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_MECHANISMS = {
    "abrupt_mean_shift",
    "gradual_degradation",
    "variance_instability",
    "sensor_fault",
    "recipe_product_mix_drift",
    "contextual_anomaly",
}

REQUIRED_ANOMALY_RATES = {0.01, 0.03, 0.05, 0.07}
REQUIRED_LABEL_DELAYS = {12, 24, 48}

REQUIRED_TIMELINE_RULES = {
    "alarm_before_affected_lot",
    "maintenance_delay_before_degradation",
    "recipe_change_before_mix_drift",
}

REQUIRED_OUTPUTS = {
    "lots",
    "tool_events",
    "maintenance",
    "process_changes",
    "rca_ground_truth",
}

REQUIRED_QUALITY_RULES = {
    "require_unique_lot_id",
    "require_deterministic_generation",
    "require_evidence_for_anomalous_lots",
    "require_known_root_cause_for_injected_anomalies",
    "prevent_label_availability_leakage",
    "preserve_benign_drift_cases",
}


class SyntheticV2ConfigError(ValueError):
    """Raised when the Synthetic Data V2 configuration is invalid."""


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise SyntheticV2ConfigError(
            "The configuration root must be a JSON object."
        )

    return config


def _collect_ids(
    records: Any,
    id_field: str,
    section_name: str,
    errors: list[str],
) -> list[str]:
    if not isinstance(records, list):
        errors.append(f"{section_name} must be a list.")
        return []

    identifiers: list[str] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(
                f"{section_name}[{index}] must be a JSON object."
            )
            continue

        identifier = record.get(id_field)

        if not isinstance(identifier, str) or not identifier.strip():
            errors.append(
                f"{section_name}[{index}] must contain a non-empty "
                f"'{id_field}'."
            )
            continue

        identifiers.append(identifier)

    if len(identifiers) != len(set(identifiers)):
        errors.append(
            f"{section_name} contains duplicate '{id_field}' values."
        )

    return identifiers


def validate_config(
    config: dict[str, Any],
    repo_root: Path,
    check_input_path: bool = True,
) -> None:
    errors: list[str] = []

    if config.get("schema_version") != "2.0":
        errors.append("schema_version must be '2.0'.")

    if not isinstance(config.get("dataset_name"), str):
        errors.append("dataset_name must be a string.")

    random_seed = config.get("random_seed")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        errors.append("random_seed must be an integer.")

    provenance = config.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be a JSON object.")
        provenance = {}

    if provenance.get("base_dataset") != "UCI SECOM":
        errors.append("provenance.base_dataset must be 'UCI SECOM'.")

    disclosure = str(provenance.get("disclosure", "")).lower()
    disclosure_terms = [
        "uci secom",
        "synthetic",
        "not data from a real semiconductor fab",
    ]

    for term in disclosure_terms:
        if term not in disclosure:
            errors.append(
                "provenance.disclosure must contain the phrase "
                f"'{term}'."
            )

    input_path_value = provenance.get("input_path")
    if not isinstance(input_path_value, str) or not input_path_value.strip():
        errors.append(
            "provenance.input_path must be a non-empty string."
        )
    elif check_input_path:
        input_path = repo_root / input_path_value
        if not input_path.is_file():
            errors.append(
                f"SECOM input file does not exist: {input_path}"
            )

    experiment_grid = config.get("experiment_grid")
    if not isinstance(experiment_grid, dict):
        errors.append("experiment_grid must be a JSON object.")
        experiment_grid = {}

    try:
        anomaly_rates = {
            float(value)
            for value in experiment_grid.get("anomaly_rates", [])
        }
    except (TypeError, ValueError):
        anomaly_rates = set()
        errors.append(
            "experiment_grid.anomaly_rates must contain numeric values."
        )

    if anomaly_rates != REQUIRED_ANOMALY_RATES:
        errors.append(
            "experiment_grid.anomaly_rates must be exactly "
            "[0.01, 0.03, 0.05, 0.07]."
        )

    try:
        label_delays = {
            int(value)
            for value in experiment_grid.get("label_delay_hours", [])
        }
    except (TypeError, ValueError):
        label_delays = set()
        errors.append(
            "experiment_grid.label_delay_hours must contain integers."
        )

    if label_delays != REQUIRED_LABEL_DELAYS:
        errors.append(
            "experiment_grid.label_delay_hours must be exactly "
            "[12, 24, 48]."
        )

    unseen_context = experiment_grid.get("unseen_context")
    if not isinstance(unseen_context, dict):
        errors.append(
            "experiment_grid.unseen_context must be a JSON object."
        )
        unseen_context = {}

    if unseen_context.get("enabled") is not True:
        errors.append(
            "experiment_grid.unseen_context.enabled must be true."
        )

    context = config.get("context")
    if not isinstance(context, dict):
        errors.append("context must be a JSON object.")
        context = {}

    tools = context.get("tools", [])
    if not isinstance(tools, list) or len(tools) < 2:
        errors.append("context.tools must contain at least two tools.")
        tools = []

    holdout_tools = unseen_context.get("holdout_tools", [])
    if not isinstance(holdout_tools, list) or not holdout_tools:
        errors.append(
            "unseen_context.holdout_tools must contain at least one tool."
        )
    elif not set(holdout_tools).issubset(set(tools)):
        errors.append(
            "Every holdout tool must also appear in context.tools."
        )

    for field in ("recipes", "product_families", "shifts"):
        values = context.get(field)
        if not isinstance(values, list) or not values:
            errors.append(f"context.{field} must be a non-empty list.")

    mechanisms = config.get("anomaly_mechanisms", [])
    mechanism_ids = _collect_ids(
        mechanisms,
        "id",
        "anomaly_mechanisms",
        errors,
    )

    if set(mechanism_ids) != REQUIRED_MECHANISMS:
        errors.append(
            "anomaly_mechanisms must define exactly these six IDs: "
            + ", ".join(sorted(REQUIRED_MECHANISMS))
        )

    if isinstance(mechanisms, list):
        for mechanism in mechanisms:
            if not isinstance(mechanism, dict):
                continue

            mechanism_id = mechanism.get("id", "<unknown>")

            if mechanism.get("enabled") is not True:
                errors.append(
                    f"Anomaly mechanism '{mechanism_id}' must be enabled."
                )

            parameters = mechanism.get("parameters")
            if not isinstance(parameters, dict) or not parameters:
                errors.append(
                    f"Anomaly mechanism '{mechanism_id}' must contain "
                    "non-empty parameters."
                )

    timeline_rule_ids = set(
        _collect_ids(
            config.get("timeline_rules", []),
            "id",
            "timeline_rules",
            errors,
        )
    )

    if timeline_rule_ids != REQUIRED_TIMELINE_RULES:
        errors.append(
            "timeline_rules must define alarm, maintenance-delay, and "
            "recipe-change temporal relationships."
        )

    root_causes = config.get("root_causes", [])
    root_cause_ids = _collect_ids(
        root_causes,
        "root_cause_id",
        "root_causes",
        errors,
    )

    if len(root_cause_ids) != len(REQUIRED_MECHANISMS):
        errors.append(
            "root_causes must contain one ground-truth cause for each "
            "anomaly mechanism."
        )

    covered_mechanisms: set[str] = set()

    if isinstance(root_causes, list):
        for root_cause in root_causes:
            if not isinstance(root_cause, dict):
                continue

            mechanism = root_cause.get("mechanism")
            if isinstance(mechanism, str):
                covered_mechanisms.add(mechanism)

            if not isinstance(root_cause.get("is_fault"), bool):
                errors.append(
                    "Every root cause must contain a Boolean is_fault."
                )

    if covered_mechanisms != REQUIRED_MECHANISMS:
        errors.append(
            "root_causes must cover all six anomaly mechanisms exactly."
        )

    outputs = config.get("outputs", [])
    output_ids = set(
        _collect_ids(outputs, "id", "outputs", errors)
    )

    if output_ids != REQUIRED_OUTPUTS:
        errors.append(
            "outputs must define exactly: "
            + ", ".join(sorted(REQUIRED_OUTPUTS))
        )

    if isinstance(outputs, list):
        for output in outputs:
            if not isinstance(output, dict):
                continue

            output_id = output.get("id", "<unknown>")
            output_path = output.get("path")
            required_columns = output.get("required_columns")

            if not isinstance(output_path, str) or not output_path.endswith(
                ".csv"
            ):
                errors.append(
                    f"Output '{output_id}' must have a CSV path."
                )

            if (
                not isinstance(required_columns, list)
                or not required_columns
                or len(required_columns) != len(set(required_columns))
            ):
                errors.append(
                    f"Output '{output_id}' must have unique, non-empty "
                    "required_columns."
                )

    quality_rules = config.get("quality_rules")
    if not isinstance(quality_rules, dict):
        errors.append("quality_rules must be a JSON object.")
        quality_rules = {}

    missing_quality_rules = (
        REQUIRED_QUALITY_RULES - set(quality_rules)
    )

    if missing_quality_rules:
        errors.append(
            "Missing quality rules: "
            + ", ".join(sorted(missing_quality_rules))
        )

    for rule_name in REQUIRED_QUALITY_RULES:
        if quality_rules.get(rule_name) is not True:
            errors.append(
                f"quality_rules.{rule_name} must be true."
            )

    if errors:
        formatted_errors = "\n".join(
            f"- {error}" for error in errors
        )
        raise SyntheticV2ConfigError(formatted_errors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the WaferWatch Synthetic Data V2 experiment contract."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_data_v2.json"),
        help="Path to the Synthetic Data V2 JSON configuration.",
    )
    parser.add_argument(
        "--skip-input-check",
        action="store_true",
        help="Validate the contract without checking the SECOM input file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    repo_root = Path.cwd().resolve()

    try:
        config = load_config(config_path)
        validate_config(
            config=config,
            repo_root=repo_root,
            check_input_path=not args.skip_input_check,
        )
    except FileNotFoundError:
        print("SYNTHETIC_V2_CONFIG_INVALID")
        print(f"- Configuration file not found: {config_path}")
        return 1
    except json.JSONDecodeError as error:
        print("SYNTHETIC_V2_CONFIG_INVALID")
        print(
            f"- Invalid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        )
        return 1
    except SyntheticV2ConfigError as error:
        print("SYNTHETIC_V2_CONFIG_INVALID")
        print(error)
        return 1

    experiment_grid = config["experiment_grid"]
    mechanism_count = len(config["anomaly_mechanisms"])
    output_count = len(config["outputs"])

    rates = ", ".join(
        f"{rate:.0%}"
        for rate in experiment_grid["anomaly_rates"]
    )
    delays = ", ".join(
        f"{delay}h"
        for delay in experiment_grid["label_delay_hours"]
    )

    print("SYNTHETIC_V2_CONFIG_OK")
    print(f"Config: {config_path}")
    print(f"Schema version: {config['schema_version']}")
    print(f"Random seed: {config['random_seed']}")
    print(f"Anomaly mechanisms: {mechanism_count}")
    print(f"Anomaly rates: {rates}")
    print(f"Label delays: {delays}")
    print(f"Output tables: {output_count}")
    print("Unseen-context evaluation: enabled")
    print("No synthetic data has been generated yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())