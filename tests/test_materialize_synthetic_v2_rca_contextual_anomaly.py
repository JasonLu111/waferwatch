from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.materialize_synthetic_v2_rca_evaluation import (
    RCA_CONTEXTUAL_ANOMALY_OUTPUT_FILES,
    materialize_rca_contextual_anomaly,
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
                "id": "RCA_CONTEXTUAL_ANOMALY",
                "scenario_type": "root_cause",
                "description": "Evaluate contextual RCA with paired controls.",
                "root_cause_id": "RC_RECIPE_CONTEXT_MISMATCH",
                "required_evidence_types": [
                    "tool_event",
                    "recipe_context",
                    "synthetic_lot",
                ],
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
                    "expected_lot_count": 2,
                    "expected_ground_truth_count": 2,
                    "expected_tool_event_count": 1,
                    "expected_recipe_context_count": 4,
                    "expected_evidence_bundle_count": 7,
                    "required_tool_event_evidence_id": "EVID_CONTEXT_EVENT_001",
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

    common_context = {
        "tool_id": "TOOL_04",
        "chamber_id": "TOOL_04_CH_01",
        "product_family": "LOGIC",
        "contextual_normal_recipe": "RCP_A",
        "contextual_anomalous_recipe": "RCP_B",
        "contextual_sensor_columns": "sensor_101|sensor_202",
    }
    rows = []
    for index, event_time in enumerate(
        ["2026-04-29 06:00:00", "2026-05-01 18:00:00"], start=1
    ):
        pair_id = f"CTX_PAIR_{index:03d}"
        rows.extend(
            [
                {
                    **common_context,
                    "lot_id": f"SYNV2_00{index}09",
                    "event_time": event_time,
                    "recipe_id": "RCP_B",
                    "is_synthetic_anomaly": True,
                    "anomaly_mechanism": "contextual_anomaly",
                    "root_cause_id": "RC_RECIPE_CONTEXT_MISMATCH",
                    "synthetic_evidence_id": (
                        f"EVID_CONTEXT_ANOMALY_{pair_id}"
                    ),
                    "is_contextual_control": False,
                    "contextual_pair_id": pair_id,
                    "contextual_expected_status": "anomaly",
                    "recipe_context_evidence_id": (
                        f"EVID_CONTEXT_ANOMALY_{pair_id}"
                    ),
                },
                {
                    **common_context,
                    "lot_id": f"SYNV2_CTRL_{index:03d}",
                    "event_time": event_time,
                    "recipe_id": "RCP_A",
                    "is_synthetic_anomaly": False,
                    "anomaly_mechanism": "none",
                    "root_cause_id": "",
                    "synthetic_evidence_id": "",
                    "is_contextual_control": True,
                    "contextual_pair_id": pair_id,
                    "contextual_expected_status": "normal_control",
                    "recipe_context_evidence_id": (
                        f"EVID_CONTEXT_CONTROL_{pair_id}"
                    ),
                },
            ]
        )
    pd.DataFrame(rows).to_csv(source_dir / "synthetic_secom_v2.csv", index=False)

    pd.DataFrame(
        [
            {
                "event_id": "EVT_CONTEXTUAL_001",
                "tool_id": "TOOL_04",
                "chamber_id": "TOOL_04_CH_01",
                "alarm_code": "ALARM_RECIPE_CONTEXT_MISMATCH",
                "event_type": "recipe_context_alarm",
                "start_time": "2026-04-29 05:00:00",
                "end_time": "2026-04-29 06:00:00",
                "severity": "high",
                "related_lot_id": "SYNV2_00109",
                "evidence_id": "EVID_CONTEXT_EVENT_001",
            }
        ]
    ).to_csv(source_dir / "synthetic_tool_events.csv", index=False)

    pd.DataFrame(
        [
            {
                "case_id": f"RCA_CONTEXTUAL_{index:03d}",
                "lot_id": f"SYNV2_00{index}09",
                "root_cause_id": "RC_RECIPE_CONTEXT_MISMATCH",
                "suspected_cause": "Synthetic recipe-context mismatch.",
                "recommended_action": "Compare against the matched normal recipe.",
                "outcome": "Synthetic contextual anomaly confirmed.",
                "evidence_ids": ";".join(
                    [
                        "EVID_CONTEXT_EVENT_001",
                        f"EVID_CONTEXT_ANOMALY_CTX_PAIR_{index:03d}",
                        f"EVID_CONTEXT_CONTROL_CTX_PAIR_{index:03d}",
                    ]
                ),
                "supports_abstention": False,
                "top3_acceptable_causes": "RC_RECIPE_CONTEXT_MISMATCH",
            }
            for index in [1, 2]
        ]
    ).to_csv(source_dir / "synthetic_rca_ground_truth.csv", index=False)


def test_rca_contextual_anomaly_cohort_contract(tmp_path: Path) -> None:
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

    summary = materialize_rca_contextual_anomaly(repo_root, output_dir)

    assert summary.scenario_id == "RCA_CONTEXTUAL_ANOMALY"
    assert summary.root_cause_id == "RC_RECIPE_CONTEXT_MISMATCH"
    assert summary.cohort_lot_count == 2
    assert summary.ground_truth_count == 2
    assert summary.tool_event_count == 1
    assert summary.context_evidence_type == "tool_event"
    assert summary.context_evidence_count == 1
    assert summary.recipe_context_count == 4
    assert summary.evidence_bundle_count == 7
    assert {path: _sha256(path) for path in source_paths} == before_hashes

    assert {path.name for path in output_dir.iterdir()} == set(
        RCA_CONTEXTUAL_ANOMALY_OUTPUT_FILES.values()
    )
    recipe_context = pd.read_csv(
        output_dir / RCA_CONTEXTUAL_ANOMALY_OUTPUT_FILES["recipe_context"]
    )
    assert len(recipe_context) == 4
    assert set(recipe_context["recipe_id"]) == {"RCP_A", "RCP_B"}
    assert recipe_context["is_contextual_control"].astype(str).str.lower().isin(
        {"true", "1"}
    ).sum() == 2

    evidence_bundle = pd.read_csv(
        output_dir / RCA_CONTEXTUAL_ANOMALY_OUTPUT_FILES["evidence_bundle"]
    )
    assert evidence_bundle["evidence_type"].value_counts().to_dict() == {
        "recipe_context": 4,
        "synthetic_lot": 2,
        "tool_event": 1,
    }
    assert set(evidence_bundle["evidence_id"]) == {
        "EVID_CONTEXT_EVENT_001",
        "EVID_CONTEXT_ANOMALY_CTX_PAIR_001",
        "EVID_CONTEXT_CONTROL_CTX_PAIR_001",
        "EVID_CONTEXT_ANOMALY_CTX_PAIR_002",
        "EVID_CONTEXT_CONTROL_CTX_PAIR_002",
    }
    assert evidence_bundle["evidence_id"].nunique() == 5

    generated_manifest = json.loads(
        (
            output_dir / RCA_CONTEXTUAL_ANOMALY_OUTPUT_FILES["manifest"]
        ).read_text(encoding="utf-8")
    )
    assert generated_manifest["cohort"]["expected_top_k"] == [1, 3]
    assert generated_manifest["cohort"]["expected_abstention"] is False
    assert generated_manifest["recipe_context_evidence"]["pair_count"] == 2
    assert generated_manifest["recipe_context_evidence"]["normal_control_record_count"] == 2
    assert generated_manifest["evidence_bundle"]["recipe_context_count"] == 4
    assert generated_manifest["evidence_bundle"]["evidence_id_count"] == 5
    assert generated_manifest["evidence_bundle"]["row_count"] == 7


def test_rca_contextual_anomaly_cohort_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _write_fixture_repository(repo_root)
    first_output_dir = tmp_path / "first"
    second_output_dir = tmp_path / "second"

    materialize_rca_contextual_anomaly(repo_root, first_output_dir)
    materialize_rca_contextual_anomaly(repo_root, second_output_dir)

    for filename in RCA_CONTEXTUAL_ANOMALY_OUTPUT_FILES.values():
        assert (first_output_dir / filename).read_bytes() == (
            second_output_dir / filename
        ).read_bytes()
