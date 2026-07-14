from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from src.data.materialize_synthetic_v2_benign_drift import (
    BENIGN_TOOL_REASSIGNMENT_OUTPUT_FILES,
    materialize_benign_tool_reassignment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _true_mask(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true"})


def test_benign_tool_reassignment_contract_and_rag_ground_truth(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "BENIGN_TOOL_REASSIGNMENT"

    summary = materialize_benign_tool_reassignment(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
    )

    assert summary.scenario_id == "BENIGN_TOOL_REASSIGNMENT"
    assert summary.benign_lot_count == 24
    assert summary.process_change_count == 1
    assert summary.rag_ground_truth_count == 24

    for filename in BENIGN_TOOL_REASSIGNMENT_OUTPUT_FILES.values():
        assert (output_dir / filename).is_file()

    lots = pd.read_csv(
        output_dir / BENIGN_TOOL_REASSIGNMENT_OUTPUT_FILES["lots"]
    )
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
            "tool_reassignment"
        )
    ].sort_values(["event_time", "lot_id"], kind="mergesort").reset_index(
        drop=True
    )

    pdt.assert_frame_equal(
        lots.reset_index(drop=True),
        expected_lots,
        check_dtype=False,
    )
    assert len(lots) == 24
    assert not _true_mask(lots["is_synthetic_anomaly"]).any()
    assert _true_mask(lots["is_benign_drift"]).all()
    assert lots["benign_drift_type"].eq("tool_reassignment").all()

    process_changes = pd.read_csv(
        output_dir
        / BENIGN_TOOL_REASSIGNMENT_OUTPUT_FILES["process_changes"]
    )
    assert len(process_changes) == 1
    change = process_changes.iloc[0]
    assert change["change_id"] == "CHANGE_BENIGN_TOOL_REASSIGNMENT_003"
    assert change["change_type"] == "tool_reassignment"
    assert change["evidence_id"] == (
        "EVID_BENIGN_DRIFT_TOOL_REASSIGNMENT_003"
    )

    rag_ground_truth = pd.read_csv(
        output_dir
        / BENIGN_TOOL_REASSIGNMENT_OUTPUT_FILES["rag_ground_truth"]
    )
    assert len(rag_ground_truth) == 24
    assert set(rag_ground_truth["lot_id"].astype(str)) == set(
        lots["lot_id"].astype(str)
    )
    assert rag_ground_truth["lot_id"].is_unique
    assert rag_ground_truth["scenario_id"].eq(
        "BENIGN_TOOL_REASSIGNMENT"
    ).all()
    assert rag_ground_truth["expected_decision"].eq(
        "abstain_or_no_escalation"
    ).all()
    assert _true_mask(rag_ground_truth["expected_abstention"]).all()
    assert rag_ground_truth["expected_root_cause_id"].eq(
        "RC_BENIGN_MIX_CHANGE"
    ).all()
    assert rag_ground_truth["evidence_ids"].eq(
        "EVID_BENIGN_DRIFT_TOOL_REASSIGNMENT_003"
    ).all()
    assert rag_ground_truth["source_rca_case_id"].eq(
        "RCA_BENIGN_DRIFT_003"
    ).all()

    generated_manifest = json.loads(
        (
            output_dir
            / BENIGN_TOOL_REASSIGNMENT_OUTPUT_FILES["manifest"]
        ).read_text(encoding="utf-8")
    )
    assert generated_manifest["scenario_id"] == "BENIGN_TOOL_REASSIGNMENT"
    assert generated_manifest["cohort"]["benign_lot_count"] == 24
    assert generated_manifest["cohort"]["synthetic_anomaly_count"] == 0
    assert generated_manifest["process_change_evidence"]["evidence_id"] == (
        "EVID_BENIGN_DRIFT_TOOL_REASSIGNMENT_003"
    )
    assert generated_manifest["rag_ground_truth"]["source_rca_case_id"] == (
        "RCA_BENIGN_DRIFT_003"
    )


def test_benign_tool_reassignment_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_benign_tool_reassignment(
        repo_root=REPO_ROOT,
        output_dir=first_output_dir,
    )
    materialize_benign_tool_reassignment(
        repo_root=REPO_ROOT,
        output_dir=second_output_dir,
    )

    for filename in BENIGN_TOOL_REASSIGNMENT_OUTPUT_FILES.values():
        assert (first_output_dir / filename).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()
