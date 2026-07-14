from copy import deepcopy
from pathlib import Path

import pytest

from src.data.validate_synthetic_v2_stress_test_manifest import (
    StressTestManifestValidationError,
    load_manifest,
    validate_manifest,
)


MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "synthetic_data_v2_stress_test_manifest.json"
)


def test_stress_test_manifest_is_valid() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    summary = validate_manifest(manifest)

    assert summary.scenario_count == 16
    assert summary.prevalence_rates == [0.01, 0.03, 0.05, 0.07]
    assert summary.label_delays == [12, 24, 48]
    assert summary.benign_drift_count == 3
    assert summary.root_cause_count == 5


def test_manifest_requires_all_prevalence_conditions() -> None:
    manifest = deepcopy(load_manifest(MANIFEST_PATH))
    manifest["scenarios"] = [
        scenario
        for scenario in manifest["scenarios"]
        if scenario["id"] != "PREV_07"
    ]

    with pytest.raises(StressTestManifestValidationError):
        validate_manifest(manifest)