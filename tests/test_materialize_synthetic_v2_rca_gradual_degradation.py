from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.materialize_synthetic_v2_rca_evaluation import (
    RCA_GRADUAL_DEGRADATION_OUTPUT_FILES,
    materialize_rca_gradual_degradation,
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
                "id": "RCA_GRADUAL_DEGRADATION",
                "scenario_type": "root_cause",
                "description": "Evaluate gradual-degradation RCA.",
                "root_cause_id": "RC_COMPONENT_WEAR",
                "required_evidence_types": ["maintenance", "synthetic_lot"],
                "expected_top_k": [1, 3],
                "expected_abstention": False,
                "materialization": {
                    "mode": "rca_evaluation_cohort",
                    "source_lots_path": "data/synthetic/v2/synthetic_secom_v2.csv",
                    "source_maintenance_path": "data/synthetic/v2/synthetic_maintenance.csv",
                    "source_rca_ground_truth_path": "data/synthetic/v2/synthetic_rca_ground_truth.csv",
                    "expected_lot_count": 2,
                    "expected_ground_truth_count": 2,
                    "expected_maintenance_count": 1,
                    "expected_evidence_bundle_count": 3,
                    "required_maintenance_evidence_id": "EVID_MAINT_DELAY_GRADUAL_001",
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
                "lot_id": "SYNV2_00019",
                "event_time": "2026-01-04 06:00:00",
                "tool_id": "TOOL_05",
                "chamber_id": "TOOL_05_CH_02",
                "recipe_id": "RCP_A",
                "product_family": "LOGIC",
                "anomaly_mechanism": "gradual_degradation",
                "root_cause_id": "RC_COMPONENT_WEAR",
                "is_synthetic_anomaly": True,
                "synthetic_evidence_id": "EVID_DEGRADATION_SYNV2_00019",
                "injected_sensor_columns": "sensor_001|sensor_002",
            },
            {
                "lot_id": "SYNV2_00038",
                "event_time": "2026-01-07 10:00:00",
                "tool_id": "TOOL_05",
                "chamber_id": "TOOL_05_CH_02",
                "recipe_id": "RCP_B",
                "product_family": "LOGIC",
                "anomaly_mechanism": "gradual_degradation",
                "root_cause_id": "RC_COMPONENT_WEAR",
                "is_synthetic_anomaly": True,
                "synthetic_evidence_id": "EVID_DEGRADATION_SYNV2_00038",
                "injected_sensor_columns": "sensor_001|sensor_003",
            },
            {
                "lot_id": "SYNV2_00039",
                "event_time": "2026-01-07 14:00:00",
                "tool_id": "TOOL_01",
                "chamber_id": "TOOL_01_CH_01",
                "recipe_id": "RCP_A",
                "product_family": "MEMORY",
                "anomaly_mechanism": "none",
                "root_cause_id": "",
                "is_synthetic_anomaly": False,
                "synthetic_evidence_id": "",
                "injected_sensor_columns": "",
            },
        ]
    ).to_csv(source_dir / "synthetic_secom_v2.csv", index=False)

    pd.DataFrame(
        [
            {
                "maintenance_id": "MAINT_GRADUAL_001",
                "tool_id": "TOOL_05",
                "chamber_id": "TOOL_05_CH_02",
                "scheduled_at": "2025-12-30 06:00:00",
                "performed_at": "2026-01-04 06:00:00",
                "delay_days": 5,
                "maintenance_type": "preventive_inspection",
                "component": "pressure_control_valve",
                "replacement_performed": False,
                "related_lot_id": "SYNV2_00019",
                "evidence_id": "EVID_MAINT_DELAY_GRADUAL_001",
            },
            {
                "maintenance_id": "MAINT_OTHER_001",
                "tool_id": "TOOL_01",
                "chamber_id": "TOOL_01_CH_01",
                "scheduled_at": "2026-01-05 06:00:00",
                "performed_at": "2026-01-05 06:00:00",
                "delay_days": 0,
                "maintenance_type": "other",
                "component": "other_component",
                "replacement_performed": False,
                "related_lot_id": "SYNV2_00039",
                "evidence_id": "EVID_OTHER_001",
            },
        ]
    ).to_csv(source_dir / "synthetic_maintenance.csv", index=False)

    pd.DataFrame(
        [
            {
                "case_id": "RCA_GRADUAL_001",
                "lot_id": "SYNV2_00019",
                "root_cause_id": "RC_COMPONENT_WEAR",
                "evidence_ids": "EVID_MAINT_DELAY_GRADUAL_001;EVID_DEGRADATION_SYNV2_00019",
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_COMPONENT_WEAR",
            },
            {
                "case_id": "RCA_GRADUAL_002",
                "lot_id": "SYNV2_00038",
                "root_cause_id": "RC_COMPONENT_WEAR",
                "evidence_ids": "EVID_MAINT_DELAY_GRADUAL_001;EVID_DEGRADATION_SYNV2_00038",
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_COMPONENT_WEAR",
            },
            {
                "case_id": "RCA_OTHER_001",
                "lot_id": "SYNV2_00039",
                "root_cause_id": "RC_OTHER",
                "evidence_ids": "EVID_OTHER_001",
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_OTHER",
            },
        ]
    ).to_csv(source_dir / "synthetic_rca_ground_truth.csv", index=False)


def test_rca_gradual_degradation_cohort_contract(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_fixture_repository(repo_root)
    output_dir = tmp_path / "cohort"
    source_paths = [
        repo_root / "data" / "synthetic" / "v2" / filename
        for filename in [
            "synthetic_secom_v2.csv",
            "synthetic_maintenance.csv",
            "synthetic_rca_ground_truth.csv",
        ]
    ]
    before_hashes = {path: _sha256(path) for path in source_paths}

    summary = materialize_rca_gradual_degradation(repo_root, output_dir)

    assert summary.scenario_id == "RCA_GRADUAL_DEGRADATION"
    assert summary.root_cause_id == "RC_COMPONENT_WEAR"
    assert summary.cohort_lot_count == 2
    assert summary.ground_truth_count == 2
    assert summary.context_evidence_type == "maintenance"
    assert summary.context_evidence_count == 1
    assert summary.evidence_bundle_count == 3
    assert {path: _sha256(path) for path in source_paths} == before_hashes

    assert {path.name for path in output_dir.iterdir()} == set(
        RCA_GRADUAL_DEGRADATION_OUTPUT_FILES.values()
    )
    evidence_bundle = pd.read_csv(
        output_dir / RCA_GRADUAL_DEGRADATION_OUTPUT_FILES["evidence_bundle"]
    )
    assert list(evidence_bundle["evidence_type"]) == [
        "maintenance",
        "synthetic_lot",
        "synthetic_lot",
    ]
    assert set(evidence_bundle["evidence_id"]) == {
        "EVID_MAINT_DELAY_GRADUAL_001",
        "EVID_DEGRADATION_SYNV2_00019",
        "EVID_DEGRADATION_SYNV2_00038",
    }

    generated_manifest = json.loads(
        (
            output_dir / RCA_GRADUAL_DEGRADATION_OUTPUT_FILES["manifest"]
        ).read_text(encoding="utf-8")
    )
    assert generated_manifest["cohort"]["expected_top_k"] == [1, 3]
    assert generated_manifest["cohort"]["expected_abstention"] is False
    assert generated_manifest["context_evidence"]["evidence_type"] == "maintenance"
    assert generated_manifest["evidence_bundle"]["maintenance_count"] == 1


def test_rca_gradual_degradation_cohort_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _write_fixture_repository(repo_root)
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_rca_gradual_degradation(repo_root, first_output_dir)
    materialize_rca_gradual_degradation(repo_root, second_output_dir)

    for filename in RCA_GRADUAL_DEGRADATION_OUTPUT_FILES.values():
        assert (first_output_dir / filename).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()