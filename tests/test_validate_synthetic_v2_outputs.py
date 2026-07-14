from pathlib import Path

from src.data.validate_synthetic_v2_outputs import (
    validate_output_tables,
    validate_reproducibility,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_v2_output_contract_passes() -> None:
    summary = validate_output_tables(REPO_ROOT)

    assert summary.lot_count == 1567
    assert summary.synthetic_anomaly_count == 235
    assert summary.benign_drift_count == 72
    assert summary.rca_case_count == 238
    assert summary.mechanism_counts == {
        "abrupt_mean_shift": 47,
        "gradual_degradation": 47,
        "variance_instability": 47,
        "sensor_fault": 47,
        "contextual_anomaly": 47,
    }
    assert set(summary.output_hashes) == {
        "lots",
        "tool_events",
        "maintenance",
        "process_changes",
        "rca_ground_truth",
    }


def test_synthetic_v2_outputs_are_byte_reproducible() -> None:
    validate_reproducibility(REPO_ROOT)