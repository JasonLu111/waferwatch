from pathlib import Path

from src.data.materialize_synthetic_v2_scenario import (
    COHORT_OUTPUT_FILES,
    materialize_scenario,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prev_01_cohort_matches_materialization_contract(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "PREV_01"

    summary = materialize_scenario(
        repo_root=REPO_ROOT,
        scenario_id="PREV_01",
        output_dir=output_dir,
    )

    assert summary.scenario_id == "PREV_01"
    assert summary.cohort_size == 1000
    assert summary.synthetic_anomaly_count == 10
    assert summary.achieved_anomaly_rate == 0.01
    assert summary.mechanism_counts == {
        "abrupt_mean_shift": 2,
        "gradual_degradation": 2,
        "variance_instability": 2,
        "sensor_fault": 2,
        "contextual_anomaly": 2,
    }
    assert summary.label_delay_counts == {
        "12": 334,
        "24": 333,
        "48": 333,
    }

    for filename in COHORT_OUTPUT_FILES.values():
        assert (output_dir / filename).exists()


def test_prev_01_materialization_is_reproducible(
    tmp_path: Path,
) -> None:
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_scenario(
        repo_root=REPO_ROOT,
        scenario_id="PREV_01",
        output_dir=first_output_dir,
    )
    materialize_scenario(
        repo_root=REPO_ROOT,
        scenario_id="PREV_01",
        output_dir=second_output_dir,
    )

    for filename in COHORT_OUTPUT_FILES.values():
        assert (
            first_output_dir / filename
        ).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()