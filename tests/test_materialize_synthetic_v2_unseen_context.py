from pathlib import Path

from src.data.materialize_synthetic_v2_unseen_context import (
    OUTPUT_FILES,
    materialize_unseen_tool_chamber,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unseen_tool_chamber_overlay_contract(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "UNSEEN_TOOL_CHAMBER"

    summary = materialize_unseen_tool_chamber(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
    )

    assert summary.training_lot_count == 1374
    assert summary.evaluation_lot_count == 193
    assert summary.synthetic_anomaly_count == 6
    assert summary.achieved_anomaly_rate == 6 / 193
    assert summary.mechanism_counts == {
        "abrupt_mean_shift": 1,
        "gradual_degradation": 1,
        "variance_instability": 1,
        "sensor_fault": 2,
        "contextual_anomaly": 1,
    }

    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).exists()


def test_unseen_tool_chamber_overlay_is_reproducible(
    tmp_path: Path,
) -> None:
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_unseen_tool_chamber(
        repo_root=REPO_ROOT,
        output_dir=first_output_dir,
    )
    materialize_unseen_tool_chamber(
        repo_root=REPO_ROOT,
        output_dir=second_output_dir,
    )

    for filename in OUTPUT_FILES.values():
        assert (
            first_output_dir / filename
        ).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()