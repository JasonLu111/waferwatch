from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.materialize_synthetic_v2_rca_evaluation import (
    RCA_SENSOR_FAULT_OUTPUT_FILES,
    materialize_rca_sensor_fault,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_repository(repo_root: Path) -> None:
    (repo_root / "configs").mkdir(parents=True)
    source_dir = repo_root / "data" / "synthetic" / "v2"
    source_dir.mkdir(parents=True)

    event_ids = [
        "EVID_SENSOR_EVENT_STUCK_AT_001",
        "EVID_SENSOR_EVENT_DROPOUT_002",
        "EVID_SENSOR_EVENT_MISSING_BURST_003",
        "EVID_SENSOR_EVENT_CALIBRATION_BIAS_004",
    ]
    manifest = {
        "scenarios": [
            {
                "id": "RCA_SENSOR_FAULT",
                "scenario_type": "root_cause",
                "description": "Evaluate sensor-fault RCA.",
                "root_cause_id": "RC_SENSOR_HARDWARE",
                "required_evidence_types": ["tool_event", "synthetic_lot"],
                "expected_top_k": [1, 3],
                "expected_abstention": False,
                "materialization": {
                    "mode": "rca_evaluation_cohort",
                    "source_lots_path": "data/synthetic/v2/synthetic_secom_v2.csv",
                    "source_tool_events_path": (
                        "data/synthetic/v2/synthetic_tool_events.csv"
                    ),
                    "source_rca_ground_truth_path": (
                        "data/synthetic/v2/synthetic_rca_ground_truth.csv"
                    ),
                    "expected_lot_count": 4,
                    "expected_ground_truth_count": 4,
                    "expected_tool_event_count": 4,
                    "expected_evidence_bundle_count": 8,
                    "required_tool_event_evidence_ids": event_ids,
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

    lot_ids = [
        "SYNV2_00066",
        "SYNV2_00250",
        "SYNV2_00425",
        "SYNV2_00605",
    ]
    lot_times = [
        "2026-01-12 02:00:00",
        "2026-02-11 18:00:00",
        "2026-03-12 22:00:00",
        "2026-04-11 22:00:00",
    ]
    fault_modes = [
        "stuck_at",
        "dropout",
        "missing_burst",
        "calibration_bias",
    ]
    lot_evidence_ids = [
        "EVID_SENSOR_STUCK_AT_SYNV2_00066",
        "EVID_SENSOR_DROPOUT_SYNV2_00250",
        "EVID_SENSOR_MISSING_BURST_SYNV2_00425",
        "EVID_SENSOR_CALIBRATION_BIAS_SYNV2_00605",
    ]
    pd.DataFrame(
        [
            {
                "lot_id": lot_id,
                "event_time": event_time,
                "tool_id": "TOOL_06",
                "chamber_id": "TOOL_06_CH_01",
                "recipe_id": "RCP_A",
                "product_family": "LOGIC",
                "anomaly_mechanism": "sensor_fault",
                "root_cause_id": "RC_SENSOR_HARDWARE",
                "is_synthetic_anomaly": True,
                "synthetic_evidence_id": lot_evidence_id,
                "injected_sensor_columns": "sensor_101|sensor_202",
                "sensor_fault_mode": fault_mode,
            }
            for lot_id, event_time, lot_evidence_id, fault_mode in zip(
                lot_ids, lot_times, lot_evidence_ids, fault_modes, strict=True
            )
        ]
    ).to_csv(source_dir / "synthetic_secom_v2.csv", index=False)

    event_starts = [
        "2026-01-12 01:00:00",
        "2026-02-11 17:00:00",
        "2026-03-12 21:00:00",
        "2026-04-11 21:00:00",
    ]
    alarm_codes = [
        "ALARM_SENSOR_STUCK",
        "ALARM_SENSOR_DROPOUT",
        "ALARM_SENSOR_MISSING_BURST",
        "ALARM_SENSOR_CALIBRATION_BIAS",
    ]
    pd.DataFrame(
        [
            {
                "event_id": f"EVT_SENSOR_{index:03d}",
                "tool_id": "TOOL_06",
                "chamber_id": "TOOL_06_CH_01",
                "alarm_code": alarm_code,
                "event_type": "sensor_fault_alarm",
                "start_time": start_time,
                "end_time": start_time,
                "severity": "high",
                "related_lot_id": lot_id,
                "evidence_id": event_id,
            }
            for index, (event_id, lot_id, start_time, alarm_code) in enumerate(
                zip(event_ids, lot_ids, event_starts, alarm_codes, strict=True),
                start=1,
            )
        ]
    ).to_csv(source_dir / "synthetic_tool_events.csv", index=False)

    pd.DataFrame(
        [
            {
                "case_id": f"RCA_SENSOR_{index:03d}",
                "lot_id": lot_id,
                "root_cause_id": "RC_SENSOR_HARDWARE",
                "suspected_cause": "Synthetic sensor hardware fault.",
                "recommended_action": "Inspect the affected sensor hardware.",
                "outcome": "Synthetic fault confirmed.",
                "evidence_ids": f"{event_id};{lot_evidence_id}",
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_SENSOR_HARDWARE",
            }
            for index, (lot_id, event_id, lot_evidence_id) in enumerate(
                zip(lot_ids, event_ids, lot_evidence_ids, strict=True), start=1
            )
        ]
    ).to_csv(source_dir / "synthetic_rca_ground_truth.csv", index=False)


def test_rca_sensor_fault_cohort_contract(tmp_path: Path) -> None:
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

    summary = materialize_rca_sensor_fault(repo_root, output_dir)

    assert summary.scenario_id == "RCA_SENSOR_FAULT"
    assert summary.root_cause_id == "RC_SENSOR_HARDWARE"
    assert summary.cohort_lot_count == 4
    assert summary.ground_truth_count == 4
    assert summary.tool_event_count == 4
    assert summary.context_evidence_type == "tool_event"
    assert summary.context_evidence_count == 4
    assert summary.evidence_bundle_count == 8
    assert {path: _sha256(path) for path in source_paths} == before_hashes

    assert {path.name for path in output_dir.iterdir()} == set(
        RCA_SENSOR_FAULT_OUTPUT_FILES.values()
    )
    evidence_bundle = pd.read_csv(
        output_dir / RCA_SENSOR_FAULT_OUTPUT_FILES["evidence_bundle"]
    )
    assert list(evidence_bundle["evidence_type"].head(4)) == [
        "tool_event",
        "tool_event",
        "tool_event",
        "tool_event",
    ]
    assert set(evidence_bundle["evidence_id"]) == {
        "EVID_SENSOR_EVENT_STUCK_AT_001",
        "EVID_SENSOR_EVENT_DROPOUT_002",
        "EVID_SENSOR_EVENT_MISSING_BURST_003",
        "EVID_SENSOR_EVENT_CALIBRATION_BIAS_004",
        "EVID_SENSOR_STUCK_AT_SYNV2_00066",
        "EVID_SENSOR_DROPOUT_SYNV2_00250",
        "EVID_SENSOR_MISSING_BURST_SYNV2_00425",
        "EVID_SENSOR_CALIBRATION_BIAS_SYNV2_00605",
    }
    stuck_at_event = evidence_bundle.loc[
        evidence_bundle["evidence_id"].eq("EVID_SENSOR_EVENT_STUCK_AT_001")
    ].iloc[0]
    assert stuck_at_event["event_time"] == "2026-01-12 01:00:00"

    generated_manifest = json.loads(
        (
            output_dir / RCA_SENSOR_FAULT_OUTPUT_FILES["manifest"]
        ).read_text(encoding="utf-8")
    )
    assert generated_manifest["cohort"]["expected_top_k"] == [1, 3]
    assert generated_manifest["cohort"]["expected_abstention"] is False
    assert generated_manifest["context_evidence"]["required_evidence_ids"] == sorted(
        [
            "EVID_SENSOR_EVENT_STUCK_AT_001",
            "EVID_SENSOR_EVENT_DROPOUT_002",
            "EVID_SENSOR_EVENT_MISSING_BURST_003",
            "EVID_SENSOR_EVENT_CALIBRATION_BIAS_004",
        ]
    )


def test_rca_sensor_fault_cohort_is_byte_reproducible(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_fixture_repository(repo_root)
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_rca_sensor_fault(repo_root, first_output_dir)
    materialize_rca_sensor_fault(repo_root, second_output_dir)

    for filename in RCA_SENSOR_FAULT_OUTPUT_FILES.values():
        assert (first_output_dir / filename).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()