from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.materialize_synthetic_v2_rca_evaluation import (
    RCA_VARIANCE_INSTABILITY_OUTPUT_FILES,
    materialize_rca_variance_instability,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_repository(repo_root: Path) -> None:
    (repo_root / "configs").mkdir(parents=True)
    source_dir = repo_root / "data" / "synthetic" / "v2"
    source_dir.mkdir(parents=True)

    manifest = {
        "scenarios": [
            {
                "id": "RCA_VARIANCE_INSTABILITY",
                "scenario_type": "root_cause",
                "description": "Evaluate variance-instability RCA.",
                "root_cause_id": "RC_PROCESS_VARIABILITY",
                "required_evidence_types": ["tool_event", "synthetic_lot"],
                "expected_top_k": [1, 3],
                "expected_abstention": False,
                "materialization": {
                    "mode": "rca_evaluation_cohort",
                    "source_lots_path": "data/synthetic/v2/synthetic_secom_v2.csv",
                    "source_tool_events_path": "data/synthetic/v2/synthetic_tool_events.csv",
                    "source_rca_ground_truth_path": "data/synthetic/v2/synthetic_rca_ground_truth.csv",
                    "expected_lot_count": 2,
                    "expected_ground_truth_count": 2,
                    "expected_tool_event_count": 1,
                    "expected_evidence_bundle_count": 3,
                    "required_tool_event_evidence_id": "EVID_VARIANCE_EVENT_001",
                    "preserve_core_v2_outputs": True,
                    "disclaimer": "Research-only synthetic cohort.",
                },
            }
        ]
    }
    (repo_root / "configs" / "synthetic_data_v2_stress_test_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "lot_id": "SYNV2_00027",
                "event_time": "2026-01-05 14:00:00",
                "tool_id": "TOOL_07",
                "chamber_id": "TOOL_07_CH_01",
                "recipe_id": "RCP_A",
                "product_family": "LOGIC",
                "anomaly_mechanism": "variance_instability",
                "root_cause_id": "RC_PROCESS_VARIABILITY",
                "is_synthetic_anomaly": True,
                "synthetic_evidence_id": "EVID_VARIANCE_SYNV2_00027",
            },
            {
                "lot_id": "SYNV2_00056",
                "event_time": "2026-01-10 10:00:00",
                "tool_id": "TOOL_07",
                "chamber_id": "TOOL_07_CH_01",
                "recipe_id": "RCP_B",
                "product_family": "LOGIC",
                "anomaly_mechanism": "variance_instability",
                "root_cause_id": "RC_PROCESS_VARIABILITY",
                "is_synthetic_anomaly": True,
                "synthetic_evidence_id": "EVID_VARIANCE_SYNV2_00056",
            },
            {
                "lot_id": "SYNV2_00057",
                "event_time": "2026-01-10 14:00:00",
                "tool_id": "TOOL_01",
                "chamber_id": "TOOL_01_CH_01",
                "recipe_id": "RCP_A",
                "product_family": "MEMORY",
                "anomaly_mechanism": "none",
                "root_cause_id": "",
                "is_synthetic_anomaly": False,
                "synthetic_evidence_id": "",
            },
        ]
    ).to_csv(source_dir / "synthetic_secom_v2.csv", index=False)

    pd.DataFrame(
        [
            {
                "event_id": "EVT_VARIANCE_001",
                "tool_id": "TOOL_07",
                "chamber_id": "TOOL_07_CH_01",
                "alarm_code": "ALARM_PROCESS_VARIABILITY",
                "event_type": "process_variability_warning",
                "start_time": "2026-01-05 13:00:00",
                "end_time": "2026-01-05 13:30:00",
                "severity": "medium",
                "related_lot_id": "SYNV2_00027",
                "evidence_id": "EVID_VARIANCE_EVENT_001",
            },
            {
                "event_id": "EVT_OTHER_001",
                "tool_id": "TOOL_01",
                "chamber_id": "TOOL_01_CH_01",
                "alarm_code": "ALARM_OTHER",
                "event_type": "other",
                "start_time": "2026-01-10 13:00:00",
                "end_time": "2026-01-10 13:30:00",
                "severity": "low",
                "related_lot_id": "SYNV2_00057",
                "evidence_id": "EVID_OTHER_001",
            },
        ]
    ).to_csv(source_dir / "synthetic_tool_events.csv", index=False)

    pd.DataFrame(
        [
            {
                "case_id": "RCA_VARIANCE_001",
                "lot_id": "SYNV2_00027",
                "root_cause_id": "RC_PROCESS_VARIABILITY",
                "evidence_ids": "EVID_VARIANCE_EVENT_001;EVID_VARIANCE_SYNV2_00027",
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_PROCESS_VARIABILITY",
            },
            {
                "case_id": "RCA_VARIANCE_002",
                "lot_id": "SYNV2_00056",
                "root_cause_id": "RC_PROCESS_VARIABILITY",
                "evidence_ids": "EVID_VARIANCE_EVENT_001;EVID_VARIANCE_SYNV2_00056",
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_PROCESS_VARIABILITY",
            },
            {
                "case_id": "RCA_OTHER_001",
                "lot_id": "SYNV2_00057",
                "root_cause_id": "RC_OTHER",
                "evidence_ids": "EVID_OTHER_001",
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_OTHER",
            },
        ]
    ).to_csv(source_dir / "synthetic_rca_ground_truth.csv", index=False)


def test_rca_variance_instability_cohort_contract(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_fixture_repository(repo_root)
    output_dir = tmp_path / "cohort"
    source_paths = [
        repo_root / "data" / "synthetic" / "v2" / filename
        for filename in [
            "synthetic_secom_v2.csv",
            "synthetic_tool_events.csv",
            "synthetic_rca_ground_truth.csv",
        ]
    ]
    before_hashes = {path: _sha256(path) for path in source_paths}

    summary = materialize_rca_variance_instability(repo_root, output_dir)

    assert summary.scenario_id == "RCA_VARIANCE_INSTABILITY"
    assert summary.root_cause_id == "RC_PROCESS_VARIABILITY"
    assert summary.cohort_lot_count == 2
    assert summary.ground_truth_count == 2
    assert summary.tool_event_count == 1
    assert summary.context_evidence_type == "tool_event"
    assert summary.context_evidence_count == 1
    assert summary.evidence_bundle_count == 3
    assert {path: _sha256(path) for path in source_paths} == before_hashes

    assert {path.name for path in output_dir.iterdir()} == set(
        RCA_VARIANCE_INSTABILITY_OUTPUT_FILES.values()
    )
    evidence_bundle = pd.read_csv(
        output_dir / RCA_VARIANCE_INSTABILITY_OUTPUT_FILES["evidence_bundle"]
    )
    assert list(evidence_bundle["evidence_type"]) == [
        "tool_event",
        "synthetic_lot",
        "synthetic_lot",
    ]
    assert evidence_bundle.loc[0, "event_time"] == "2026-01-05 13:00:00"
    assert set(evidence_bundle["evidence_id"]) == {
        "EVID_VARIANCE_EVENT_001",
        "EVID_VARIANCE_SYNV2_00027",
        "EVID_VARIANCE_SYNV2_00056",
    }

    generated_manifest = json.loads(
        (
            output_dir / RCA_VARIANCE_INSTABILITY_OUTPUT_FILES["manifest"]
        ).read_text(encoding="utf-8")
    )
    assert generated_manifest["cohort"]["expected_top_k"] == [1, 3]
    assert generated_manifest["cohort"]["expected_abstention"] is False
    assert generated_manifest["context_evidence"]["required_evidence_id"] == (
        "EVID_VARIANCE_EVENT_001"
    )
    assert generated_manifest["evidence_bundle"]["tool_event_count"] == 1


def test_rca_variance_instability_cohort_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _write_fixture_repository(repo_root)
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_rca_variance_instability(repo_root, first_output_dir)
    materialize_rca_variance_instability(repo_root, second_output_dir)

    for filename in RCA_VARIANCE_INSTABILITY_OUTPUT_FILES.values():
        assert (first_output_dir / filename).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()