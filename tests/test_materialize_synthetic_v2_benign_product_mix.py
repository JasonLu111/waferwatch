from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.materialize_synthetic_v2_benign_drift import (
    BENIGN_PRODUCT_MIX_OUTPUT_FILES,
    materialize_benign_product_mix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_benign_product_mix_cohort_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "BENIGN_PRODUCT_MIX"

    summary = materialize_benign_product_mix(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
    )

    assert summary.scenario_id == "BENIGN_PRODUCT_MIX"
    assert summary.benign_lot_count == 24
    assert summary.process_change_count == 1
    assert summary.rag_ground_truth_count == 24

    lots = pd.read_csv(output_dir / BENIGN_PRODUCT_MIX_OUTPUT_FILES["lots"])
    process_changes = pd.read_csv(
        output_dir / BENIGN_PRODUCT_MIX_OUTPUT_FILES["process_changes"]
    )
    rag_ground_truth = pd.read_csv(
        output_dir / BENIGN_PRODUCT_MIX_OUTPUT_FILES["rag_ground_truth"]
    )
    generated_manifest = json.loads(
        (
            output_dir / BENIGN_PRODUCT_MIX_OUTPUT_FILES["manifest"]
        ).read_text(encoding="utf-8")
    )

    assert len(lots) == 24
    assert lots["is_synthetic_anomaly"].eq(0).all()
    assert lots["is_benign_drift"].eq(1).all()
    assert lots["benign_drift_type"].eq("product_mix_change").all()

    assert len(process_changes) == 1
    change = process_changes.iloc[0]
    assert change["change_id"] == "CHANGE_BENIGN_PRODUCT_MIX_CHANGE_002"
    assert change["change_type"] == "product_mix_change"
    assert change["evidence_id"] == "EVID_BENIGN_DRIFT_PRODUCT_MIX_CHANGE_002"

    assert len(rag_ground_truth) == 24
    assert set(rag_ground_truth["lot_id"]) == set(lots["lot_id"])
    assert rag_ground_truth["expected_decision"].eq(
        "abstain_or_no_escalation"
    ).all()
    assert rag_ground_truth["expected_abstention"].eq(True).all()
    assert rag_ground_truth["expected_root_cause_id"].eq(
        "RC_BENIGN_MIX_CHANGE"
    ).all()
    assert rag_ground_truth["evidence_ids"].eq(
        "EVID_BENIGN_DRIFT_PRODUCT_MIX_CHANGE_002"
    ).all()
    assert rag_ground_truth["source_rca_case_id"].eq(
        "RCA_BENIGN_DRIFT_002"
    ).all()

    assert generated_manifest["scenario_id"] == "BENIGN_PRODUCT_MIX"
    assert generated_manifest["benign_drift"]["lot_count"] == 24
    assert generated_manifest["process_change"]["row_count"] == 1
    assert generated_manifest["rag_ground_truth"]["row_count"] == 24


def test_benign_product_mix_is_byte_reproducible(tmp_path: Path) -> None:
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_benign_product_mix(
        repo_root=REPO_ROOT,
        output_dir=first_output_dir,
    )
    materialize_benign_product_mix(
        repo_root=REPO_ROOT,
        output_dir=second_output_dir,
    )

    for filename in BENIGN_PRODUCT_MIX_OUTPUT_FILES.values():
        assert (first_output_dir / filename).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()
