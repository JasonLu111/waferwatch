import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.data.materialize_synthetic_v2_benign_drift import (
    OUTPUT_FILES,
    RAG_GROUND_TRUTH_COLUMNS,
    materialize_benign_recipe_mix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _true_mask(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "1.0", "true", "yes"})
    )


def test_benign_recipe_mix_contract_and_rag_ground_truth(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "BENIGN_RECIPE_MIX"

    summary = materialize_benign_recipe_mix(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
    )

    assert summary.scenario_id == "BENIGN_RECIPE_MIX"
    assert summary.benign_lot_count == 24
    assert summary.process_change_count == 1
    assert summary.rag_ground_truth_count == 24

    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).is_file()

    lots = pd.read_csv(output_dir / OUTPUT_FILES["lots"])
    source_lots = pd.read_csv(
        REPO_ROOT
        / "data"
        / "synthetic"
        / "v2"
        / "synthetic_secom_v2.csv"
    )

    expected_lots = source_lots.loc[
        _true_mask(source_lots["is_benign_drift"])
        & source_lots["benign_drift_type"].astype(str).eq(
            "recipe_mix_change"
        )
    ].sort_values(
        ["event_time", "lot_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    pdt.assert_frame_equal(
        lots.reset_index(drop=True),
        expected_lots,
        check_dtype=False,
    )

    assert len(lots) == 24
    assert not _true_mask(lots["is_synthetic_anomaly"]).any()
    assert _true_mask(lots["is_benign_drift"]).all()
    assert lots["benign_drift_type"].eq(
        "recipe_mix_change"
    ).all()

    process_changes = pd.read_csv(
        output_dir / OUTPUT_FILES["process_changes"]
    )

    assert len(process_changes) == 1
    assert process_changes.iloc[0]["change_id"] == (
        "CHANGE_BENIGN_RECIPE_MIX_CHANGE_001"
    )
    assert process_changes.iloc[0]["change_type"] == (
        "recipe_mix_change"
    )
    assert process_changes.iloc[0]["evidence_id"] == (
        "EVID_BENIGN_DRIFT_RECIPE_MIX_CHANGE_001"
    )

    rag_ground_truth = pd.read_csv(
        output_dir / OUTPUT_FILES["rag_ground_truth"]
    )

    assert list(rag_ground_truth.columns) == (
        RAG_GROUND_TRUTH_COLUMNS
    )
    assert len(rag_ground_truth) == 24
    assert set(rag_ground_truth["lot_id"].astype(str)) == set(
        lots["lot_id"].astype(str)
    )
    assert rag_ground_truth["lot_id"].is_unique
    assert rag_ground_truth["scenario_id"].eq(
        "BENIGN_RECIPE_MIX"
    ).all()
    assert rag_ground_truth["ground_truth_label"].eq(
        "benign_distribution_shift"
    ).all()
    assert rag_ground_truth["expected_decision"].eq(
        "abstain_or_no_escalation"
    ).all()
    assert _true_mask(
        rag_ground_truth["expected_abstention"]
    ).all()
    assert rag_ground_truth["expected_root_cause_id"].eq(
        "RC_BENIGN_MIX_CHANGE"
    ).all()
    assert rag_ground_truth["evidence_ids"].eq(
        "EVID_BENIGN_DRIFT_RECIPE_MIX_CHANGE_001"
    ).all()
    assert rag_ground_truth["evidence_type"].eq(
        "process_change"
    ).all()
    assert rag_ground_truth["source_rca_case_id"].eq(
        "RCA_BENIGN_DRIFT_001"
    ).all()
    assert not _true_mask(
        rag_ground_truth["is_synthetic_anomaly"]
    ).any()
    assert _true_mask(
        rag_ground_truth["is_benign_drift"]
    ).all()

    generated_manifest = json.loads(
        (output_dir / OUTPUT_FILES["manifest"]).read_text(
            encoding="utf-8"
        )
    )

    assert generated_manifest["scenario_id"] == (
        "BENIGN_RECIPE_MIX"
    )
    assert generated_manifest["cohort"]["benign_lot_count"] == 24
    assert generated_manifest["cohort"]["synthetic_anomaly_count"] == 0
    assert (
        generated_manifest["process_change_evidence"]["evidence_id"]
        == "EVID_BENIGN_DRIFT_RECIPE_MIX_CHANGE_001"
    )
    assert (
        generated_manifest["rag_ground_truth"]["expected_abstention"]
        is True
    )


def test_benign_recipe_mix_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_benign_recipe_mix(
        repo_root=REPO_ROOT,
        output_dir=first_output_dir,
    )
    materialize_benign_recipe_mix(
        repo_root=REPO_ROOT,
        output_dir=second_output_dir,
    )

    for filename in OUTPUT_FILES.values():
        assert (
            first_output_dir / filename
        ).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()